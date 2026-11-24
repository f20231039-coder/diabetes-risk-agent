"""FastAPI service.

Public:  POST /chat, GET /config, GET /health, GET /stats, chat page at /
Admin:   POST /admin/dataset, GET /admin/jobs/{id}, GET /admin/models,
         POST /admin/models/{version}/activate, page at /admin
         (all need the X-Admin-Token header = ADMIN_TOKEN env var)

Run:  uvicorn health_agent.server:app --port 7860
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import logging
import os
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__, model_tool, training
from .agent import HealthAgent, SessionStore
from .llm import configured_llm_choices, load_env_file, make_client
from .schemas import FIELDS, AgentReply

load_env_file()
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_HOUR", "20"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "90"))
STATS_FILE = os.environ.get("STATS_FILE")  # optional path to persist anonymous counts
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class JsonFormatter(logging.Formatter):
    def format(self, record):
        base = {"ts": round(record.created, 3), "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        base.update(getattr(record, "extra_fields", {}))
        return json.dumps(base)


def setup_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


log = logging.getLogger("health_agent.api")


class Stats:
    """Anonymous usage counts only. No health values, no IPs, no message text."""

    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self.lock = threading.Lock()
        self.counts: Counter = Counter()
        # Hashes of random per-browser ids (made in the browser, not linked to a person).
        self.visitors: set[str] = set()
        self.result_visitors: set[str] = set()
        self.started = time.time()
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self.counts.update(data.get("counts", {}))
            self.visitors.update(data.get("visitors", []))
            self.result_visitors.update(data.get("result_visitors", []))

    def incr(self, visitor: str | None = None, got_result: bool = False, **keys: int) -> None:
        with self.lock:
            self.counts.update(keys)
            if visitor:
                self.visitors.add(visitor)
                if got_result:
                    self.result_visitors.add(visitor)
            if self.path:
                self.path.write_text(json.dumps({"counts": self.counts, "visitors": sorted(self.visitors), "result_visitors": sorted(self.result_visitors)}))

    def snapshot(self) -> dict:
        with self.lock:
            return {"unique_visitors": len(self.visitors), "visitors_with_a_result": len(self.result_visitors), **dict(self.counts)}


class RateLimiter:
    """Sliding one-hour window per client IP, in memory."""

    def __init__(self, per_hour: int):
        self.per_hour = per_hour
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        q = self.hits[key]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= self.per_hour:
            return False
        q.append(now)
        return True


class AgentPool:
    """One HealthAgent per (llm, prompt version, temperature); only whitelisted LLMs."""

    def __init__(self, default: HealthAgent):
        self.default = default
        self.agents: dict[tuple, HealthAgent] = {}
        self.choices = {c["id"]: c for c in configured_llm_choices()}

    def get(self, llm: str | None, prompt_version: str | None, temperature: float | None) -> HealthAgent:
        if not llm and not prompt_version and temperature is None:
            return self.default
        if llm and llm not in self.choices:
            raise HTTPException(status_code=400, detail=f"LLM {llm!r} is not available on this server")
        key = (llm or "default", prompt_version, temperature)
        if key not in self.agents:
            client = self.default.client
            if llm or temperature is not None:
                c = self.choices.get(llm) if llm else None
                provider = c["provider"] if c else None
                model = c["model"] if c and c["provider"] != "rules" else None
                client = make_client(provider, model, temperature=temperature or 0.0)
            self.agents[key] = HealthAgent(client, prompt_version=prompt_version)
        return self.agents[key]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    model_tool.load_all()
    app.state.agent = HealthAgent(make_client())
    app.state.pool = AgentPool(app.state.agent)
    app.state.sessions = SessionStore()
    app.state.stats = Stats(STATS_FILE)
    app.state.limiter = RateLimiter(RATE_LIMIT)
    app.state.jobs = {}
    log.info("startup", extra={"extra_fields": {"llm": app.state.agent.client.name, "models": model_tool.loaded_versions()}})
    yield


app = FastAPI(title="Diabetes Risk Health-Check Agent", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def no_stale_pages(request: Request, call_next):
    """Pages and their CSS/JS must always be revalidated (ETag), so a redeploy
    shows up immediately instead of browsers reusing an old cached copy."""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/admin") or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ------------------------------------------------------------------ chat
class ChatSettings(BaseModel):
    dataset: str | None = Field(None, description="Model set, e.g. 'nhanes' or 'pima'")
    ml_model: Literal["mlp", "naive_bayes", "compare"] | None = None
    llm: str | None = Field(None, max_length=80, description="An id from GET /config")
    prompt_version: Literal["v1", "v2"] | None = None
    temperature: float | None = Field(None, ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    session_id: str | None = Field(None, max_length=64)
    message: str = Field(..., min_length=1, max_length=1000)
    visitor_id: str | None = Field(None, max_length=64, description="Random id the browser makes once; used only to count unique visitors")
    values: dict[str, float | None] | None = Field(None, description="Structured values from the form; validated like extracted ones")
    settings: ChatSettings | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: AgentReply
    latency_ms: float
    tokens: int
    llm: str


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


def apply_settings(session, settings: ChatSettings | None) -> None:
    if not settings:
        return
    if settings.dataset:
        if settings.dataset not in training.available_datasets():
            raise HTTPException(status_code=400, detail=f"dataset {settings.dataset!r} has no trained models")
        session.dataset = settings.dataset
    if settings.ml_model:
        session.prefer_compare = settings.ml_model == "compare"
        session.model_preference = None if settings.ml_model == "compare" else settings.ml_model


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    state = request.app.state
    if not state.limiter.allow(client_ip(request)):
        raise HTTPException(status_code=429, detail=f"Rate limit: {RATE_LIMIT} messages per hour. Please try later.")
    values = {k: v for k, v in (body.values or {}).items() if k in FIELDS and v is not None}
    s = body.settings
    agent = state.pool.get(s.llm if s else None, s.prompt_version if s else None, s.temperature if s else None)
    session, is_new = state.sessions.get_or_create(body.session_id)
    apply_settings(session, s)
    try:
        reply, m = await asyncio.wait_for(agent.handle(session, body.message.strip(), values=values), timeout=REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        state.stats.incr(timeouts=1)
        raise HTTPException(status_code=504, detail="The assistant took too long to answer. Please try again.")

    ok_tools = [t.name for t in reply.tool_calls if t.ok]
    visitor = hashlib.sha256(body.visitor_id.encode()).hexdigest()[:16] if body.visitor_id else None
    state.stats.incr(
        visitor=visitor,
        got_result=reply.status == "result",
        sessions=int(is_new),
        messages=1,
        form_messages=int(bool(values)),
        tool_calls=len(ok_tools),
        rejected_tool_calls=sum(not t.ok for t in reply.tool_calls),
        **{f"status_{reply.status}": 1},
        **{f"tool_{n}": 1 for n in ok_tools},
    )
    log.info(
        "chat",
        extra={
            "extra_fields": {
                "session": session.id[:8],
                "status": reply.status,
                "llm": agent.client.name,
                "dataset": reply.dataset,
                "tier": reply.tier,
                "latency_ms": round(m.latency_ms, 1),
                "llm_calls": m.llm_calls,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "tool_calls": [(t.name, t.ok) for t in reply.tool_calls],
            }
        },
    )
    return ChatResponse(session_id=session.id, reply=reply, latency_ms=round(m.latency_ms, 1), tokens=m.total_tokens, llm=agent.client.name)


@app.get("/config")
async def config(request: Request):
    """Options for the settings panel: LLMs, datasets (with tiers + metrics), fields."""
    datasets = []
    for d in training.available_datasets():
        st = model_tool.load_models(d)
        datasets.append(
            {
                "id": d,
                "label": training.DATASETS[d]["label"],
                "version": st["version"],
                "rows": st["manifest"].get("dataset_rows"),
                "tiers": model_tool.tier_summary(d),
            }
        )
    from .schemas import RANGES, REQUIRED_FOR_PREDICTION

    return {
        "default_llm": request.app.state.agent.client.name,
        "llms": list(request.app.state.pool.choices.values()),
        "datasets": datasets,
        "default_dataset": model_tool.default_dataset(),
        "fields": [{"id": f, "label": model_tool.FIELD_LABELS[f], "range": RANGES[f], "required": f in REQUIRED_FOR_PREDICTION} for f in FIELDS],
        "prompt_versions": ["v2", "v1"],
    }


@app.get("/health")
async def health(request: Request):
    loaded = model_tool.models_loaded()
    return {
        "status": "healthy" if len(loaded) == 2 else "degraded",
        "models_loaded": loaded,
        "model_versions": model_tool.loaded_versions(),
        "llm": request.app.state.agent.client.name,
        "version": __version__,
    }


@app.get("/stats")
async def stats(request: Request):
    s = request.app.state.stats
    return {"uptime_seconds": int(time.time() - s.started), "active_sessions": len(request.app.state.sessions), **s.snapshot()}


# ------------------------------------------------------------------ admin
def require_admin(request: Request) -> None:
    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        raise HTTPException(status_code=403, detail="Admin is disabled: set ADMIN_TOKEN in the server environment.")
    given = request.headers.get("x-admin-token", "")
    if not hmac.compare_digest(given.encode(), token.encode()):
        raise HTTPException(status_code=401, detail="Wrong admin token.")


def _run_training_job(app: FastAPI, job_id: str, dataset: str, force: bool) -> None:
    job = app.state.jobs[job_id]
    try:
        out = training.train_version(dataset=dataset, force=force, note=f"retrained after upload (job {job_id})")
        if out["promoted"]:
            model_tool.load_models(dataset, reload=True)
        job.update(status="done", result=out, finished=time.time())
        log.info("training job done", extra={"extra_fields": {"job": job_id, "version": out["version"], "promoted": out["promoted"]}})
    except Exception as exc:  # report, don't crash the server
        job.update(status="failed", error=str(exc), finished=time.time())
        log.error("training job failed", extra={"extra_fields": {"job": job_id, "error": str(exc)}})


@app.post("/admin/dataset")
async def upload_dataset(request: Request, dataset: str = "pima", retrain: bool = True, force: bool = False):
    """Body: raw CSV (Content-Type text/csv). Rows are validated, merged (deduplicated)
    into the dataset, then all tiers are retrained in the background. The new version
    becomes active only if its F1 does not drop (unless force=true)."""
    require_admin(request)
    if dataset not in training.DATASETS:
        raise HTTPException(status_code=400, detail=f"dataset must be one of {sorted(training.DATASETS)}")
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty body: send the CSV file as the request body.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB).")
    try:
        rows = training.validate_frame(pd.read_csv(io.BytesIO(raw)))
    except (training.DatasetError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {exc}")
    merge = training.merge_upload(rows, dataset)
    out = {"dataset": dataset, "merge": merge, "job_id": None}
    if retrain:
        job_id = uuid.uuid4().hex[:12]
        request.app.state.jobs[job_id] = {"id": job_id, "dataset": dataset, "status": "running", "started": time.time()}
        threading.Thread(target=_run_training_job, args=(request.app, job_id, dataset, force), daemon=True).start()
        out["job_id"] = job_id
    log.info("dataset uploaded", extra={"extra_fields": {"dataset": dataset, **merge}})
    return out


@app.get("/admin/jobs/{job_id}")
async def job_status(job_id: str, request: Request):
    require_admin(request)
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job


@app.get("/admin/models")
async def list_models(request: Request):
    require_admin(request)
    reg = training.read_registry()
    versions = []
    for v in reg["versions"]:
        try:
            m = training.read_manifest(v["version"])
        except FileNotFoundError:
            continue
        versions.append(
            {
                **v,
                "active": reg["active"].get(v["dataset"]) == v["version"],
                "promotion": m.get("promotion"),
                "tiers": {t: i.get("test_metrics", {}) for t, i in m["tiers"].items()},
            }
        )
    return {"active": reg["active"], "datasets": {k: {"label": d["label"]} for k, d in training.DATASETS.items()}, "versions": versions}


@app.post("/admin/models/{version}/activate")
async def activate_version(version: str, request: Request):
    require_admin(request)
    try:
        dataset = training.activate(version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    model_tool.load_models(dataset, reload=True)
    log.info("version activated", extra={"extra_fields": {"version": version, "dataset": dataset}})
    return {"dataset": dataset, "active": version}


# ------------------------------------------------------------------ pages
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/admin", include_in_schema=False)
    async def admin_page():
        return FileResponse(WEB_DIR / "admin.html")
