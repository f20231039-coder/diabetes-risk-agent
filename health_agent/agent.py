"""The agent: extraction -> validation -> LLM tool loop -> structured reply.

The LLM never runs the model directly. It proposes a tool call; the arguments
are validated with Pydantic and checked against the values the user actually
stated; only then does our code run the trained model.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import model_tool
from .llm import LLMClient, LLMError, parse_json_object
from .prompts import TOOL_SPECS, agent_system_prompt, extraction_system_prompt, extraction_user_prompt
from .schemas import (
    DISCLAIMER,
    FIELDS,
    AgentReply,
    CompareArgs,
    Extraction,
    HealthInputs,
    PredictArgs,
    ToolCallRecord,
    validate_fields,
)

log = logging.getLogger("health_agent.agent")

MAX_STEPS = 4  # LLM calls per user message in the tool loop
MAX_HISTORY = 12  # user/assistant messages kept per session
# A probability/percentage/risk-level claim; only allowed after a real tool result.
NUMERIC_RESULT_RE = re.compile(r"\d+(?:\.\d+)?\s*%|\b0\.\d{2,}\b|probabilit|sambhavna|संभावना|(?:high|low|उच्च)\s*(?:risk|जोखिम)", re.I)


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    history: list[dict] = field(default_factory=list)
    inputs: HealthInputs = field(default_factory=HealthInputs)
    model_preference: str | None = None  # "mlp" | "naive_bayes"
    prefer_compare: bool = False  # settings panel: "compare both"
    dataset: str | None = None  # settings panel: which model set (None = default)
    has_result: bool = False
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)


@dataclass
class TurnMetrics:
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    extraction_retries: int = 0
    blocked_unsupported_results: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class HealthAgent:
    def __init__(self, client: LLMClient, prompt_version: str | None = None):
        self.client = client
        self.prompt_version = prompt_version

    # ---- step 1: structured extraction ------------------------------------
    async def extract(self, message: str, session: Session, metrics: TurnMetrics) -> Extraction:
        prev = next((m["content"] for m in reversed(session.history) if m["role"] == "assistant"), None)
        messages = [
            {"role": "system", "content": extraction_system_prompt(self.prompt_version)},
            {"role": "user", "content": extraction_user_prompt(message, prev)},
        ]
        for attempt in range(2):
            res = await self.client.chat(messages, json_mode=True)
            self._count(res, metrics)
            try:
                return Extraction.model_validate(parse_json_object(res.content or ""))
            except (ValueError, ValidationError) as exc:
                # Invalid JSON / schema: show the model its error once, then give up.
                metrics.extraction_retries += 1
                log.warning("extraction parse failed (attempt %d): %s", attempt + 1, exc)
                messages += [
                    {"role": "assistant", "content": res.content or ""},
                    {"role": "user", "content": f"That was not valid JSON for the schema ({str(exc)[:200]}). Return only the JSON object."},
                ]
        return Extraction(intent="question")

    # ---- tool execution with validation + anti-invention guard ------------
    def run_tool(self, name: str, raw_args: str, effective: HealthInputs, session: Session) -> tuple[ToolCallRecord, dict]:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return self._tool_error(name, {}, "arguments were not valid JSON")
        if not isinstance(args, dict):
            return self._tool_error(name, {}, "arguments must be a JSON object")
        args = {k: v for k, v in args.items() if v is not None}

        schema = {"predict_diabetes_risk": PredictArgs, "compare_models": CompareArgs}.get(name)
        if schema is None:
            return self._tool_error(name, args, f"unknown tool {name!r}")
        try:
            parsed = schema.model_validate(args)
        except ValidationError as exc:
            return self._tool_error(name, args, "invalid arguments: " + "; ".join(e["msg"] + f" ({e['loc'][0] if e['loc'] else ''})" for e in exc.errors()))

        # Every value must match what the user actually stated (no invented values).
        for fld in FIELDS:
            given, known = getattr(parsed, fld), getattr(effective, fld)
            if given is None:
                if known is not None:
                    setattr(parsed, fld, known)  # user gave it; the LLM just left it out
                continue
            if known is None:
                return self._tool_error(name, args, f"{fld} was not stated by the user; ask for it instead of guessing")
            if abs(float(given) - float(known)) > 0.01:
                return self._tool_error(name, args, f"{fld}={given} does not match the user's value {known:g}; use KNOWN_INPUTS")
        if name == "predict_diabetes_risk" and session.model_preference and parsed.model != session.model_preference:
            parsed.model = session.model_preference

        kwargs = {k: v for k, v in parsed.model_dump().items() if v is not None}
        # The dataset comes from session settings, never from the LLM.
        fn = model_tool.predict_diabetes_risk if name == "predict_diabetes_risk" else model_tool.compare_models
        result = fn(**kwargs, dataset=session.dataset)
        return ToolCallRecord(name=name, arguments=kwargs, ok=True), result

    @staticmethod
    def _tool_error(name: str, args: dict, error: str) -> tuple[ToolCallRecord, dict]:
        return ToolCallRecord(name=name, arguments=args, ok=False, error=error), {"error": error}

    # ---- the loop ---------------------------------------------------------
    async def handle(self, session: Session, message: str, values: dict | None = None) -> tuple[AgentReply, TurnMetrics]:
        """values: structured fields from the form (validated like extracted ones, and they win)."""
        metrics = TurnMetrics()
        start = time.perf_counter()
        try:
            reply = await self._handle(session, message, metrics, values or {})
        except LLMError as exc:
            log.error("LLM failure: %s", exc)
            reply = AgentReply(
                status="error",
                known_inputs=session.inputs,
                missing_fields=session.inputs.missing(),
                explanation="Sorry, the language model is not responding right now. Please try again in a minute.",
            )
        metrics.latency_ms = (time.perf_counter() - start) * 1000
        session.updated = time.time()
        return reply, metrics

    async def _handle(self, session: Session, message: str, metrics: TurnMetrics, values: dict) -> AgentReply:
        ext = await self.extract(message, session, metrics)
        valid, errors = validate_fields({**ext.stated(), **{k: v for k, v in values.items() if v is not None}})
        new_values = sorted(valid)
        if ext.model_preference:
            session.model_preference = ext.model_preference

        refused = ext.intent in ("off_topic", "unsafe")
        # "What if" applies on top of the saved values for this turn only.
        what_if = bool(ext.intent == "what_if" and session.has_result and valid)
        merged = session.inputs.model_copy(update=valid)
        effective = HealthInputs.model_validate(merged.model_dump())
        if not what_if and not refused:
            session.inputs = effective
        missing = effective.missing()
        intent = "compare" if (session.prefer_compare and ext.intent in ("health_info", "what_if") and not missing) else ext.intent

        state = {
            "known_inputs": effective.known(),
            "missing_fields": missing,
            "invalid_values": errors,
            "intent": intent,
            "preferred_tool": "compare_models" if session.prefer_compare else "predict_diabetes_risk",
            "what_if": what_if,
            "new_values_this_turn": new_values,
            "model_preference": session.model_preference,
        }
        messages = [{"role": "system", "content": agent_system_prompt(state)}, *session.history, {"role": "user", "content": message}]
        tools = None if refused else TOOL_SPECS

        records: list[ToolCallRecord] = []
        results: list[tuple[str, dict]] = []
        final_text = ""
        for _ in range(MAX_STEPS):
            res = await self.client.chat(messages, tools=tools)
            self._count(res, metrics)
            if not res.tool_calls:
                final_text = res.content or ""
                if not results and not session.has_result and NUMERIC_RESULT_RE.search(final_text):
                    # The LLM quoted a risk number without running the model: block it.
                    metrics.blocked_unsupported_results += 1
                    if tools and not missing and not errors:
                        messages += [
                            {"role": "assistant", "content": final_text},
                            {"role": "user", "content": "[system check] You stated a risk or probability without calling a tool. Never do that. Call the tool now, or answer without any numbers."},
                        ]
                        final_text = ""
                        continue
                    final_text = ""  # fall back to a deterministic message
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": res.content or "",
                    "tool_calls": [tc.as_message_part() for tc in res.tool_calls],
                }
            )
            for tc in res.tool_calls:
                record, output = self.run_tool(tc.name, tc.arguments, effective, session)
                records.append(record)
                if record.ok:
                    results.append((tc.name, output))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(output)})
            if results and len(records) >= 3:
                tools = None  # enough tool use; force a final answer next step
        else:
            final_text = ""  # hit the step cap

        reply = self._build_reply(session, effective, missing, errors, what_if, refused, records, results, final_text)
        session.history += [{"role": "user", "content": message}, {"role": "assistant", "content": reply.explanation}]
        session.history = session.history[-MAX_HISTORY:]
        return reply

    def _build_reply(self, session, effective, missing, errors, what_if, refused, records, results, text) -> AgentReply:
        reply = AgentReply(
            status="chat",
            known_inputs=session.inputs,
            missing_fields=missing,
            invalid_fields=sorted(errors),
            what_if=what_if,
            tool_calls=records,
            explanation=text.strip(),
        )
        if results:
            name, out = results[-1]
            used = next(r for r in reversed(records) if r.ok)
            reply.status = "result"
            reply.inputs_used = HealthInputs(**{k: used.arguments.get(k) for k in FIELDS})
            reply.insulin_imputed = out["insulin_imputed"]
            if name == "compare_models":
                reply.comparison = out
                main = out["results"]["mlp"]
            else:
                main = out
            reply.risk_class, reply.probability, reply.model_used = main["risk_class"], main["probability"], main["model_used"]
            reply.model_version, reply.tier, reply.features_used = main["model_version"], main["tier"], main["features_used"]
            reply.dataset = main["dataset"]
            reply.tier_test_metrics, reply.more_values_could_help = main["tier_test_metrics"], main["more_values_could_help"]
            session.has_result = True
        elif refused:
            reply.status = "refused"
        elif errors:
            reply.status = "invalid_input"
        elif missing:
            reply.status = "need_info"

        # Deterministic guardrails on top of the prompt.
        if not reply.explanation:
            reply.explanation = self._fallback_text(reply, errors)
        high = reply.risk_class == "high" or (reply.comparison and not reply.comparison["agree"])
        if high and not re.search(r"doctor|physician|daktar|clinic|डॉक्टर|चिकित्सक", reply.explanation, re.I):
            reply.explanation += " Please consider seeing a doctor for a proper blood test (HbA1c or fasting glucose)."
        reply.explanation = reply.explanation.replace(DISCLAIMER, "").replace("**", "").strip()
        return reply

    @staticmethod
    def _fallback_text(reply: AgentReply, errors: dict) -> str:
        if reply.status == "result":
            return f"The {model_tool.MODEL_LABELS[reply.model_used]} estimates a {reply.risk_class} risk (probability {reply.probability:.0%})."
        if reply.status == "invalid_input":
            return "Some values look wrong: " + "; ".join(errors.values()) + ". Please re-check them."
        if reply.status == "need_info":
            return f"Could you tell me your {', '.join(reply.missing_fields)}?"
        if reply.status == "refused":
            return "Sorry, I can only help with diabetes-risk screening."
        return "Tell me at least your age, glucose and BMI (insulin and blood pressure help too) and I'll estimate your diabetes risk."

    @staticmethod
    def _count(res, metrics: TurnMetrics) -> None:
        metrics.llm_calls += 1
        metrics.prompt_tokens += res.prompt_tokens
        metrics.completion_tokens += res.completion_tokens


class SessionStore:
    """In-memory sessions with TTL and a size cap. Swap for Redis to scale out."""

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 5000):
        self.ttl = ttl_seconds
        self.max = max_sessions
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None) -> tuple[Session, bool]:
        self._evict()
        if session_id and session_id in self._sessions:
            return self._sessions[session_id], False
        s = Session()
        self._sessions[s.id] = s
        return s, True

    def _evict(self) -> None:
        now = time.time()
        for sid in [k for k, s in self._sessions.items() if now - s.updated > self.ttl]:
            del self._sessions[sid]
        while len(self._sessions) >= self.max:
            oldest = min(self._sessions, key=lambda k: self._sessions[k].updated)
            del self._sessions[oldest]

    def __len__(self) -> int:
        return len(self._sessions)
