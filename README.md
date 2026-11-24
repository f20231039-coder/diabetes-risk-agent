---
title: Diabetes Risk Agent
sdk: docker
app_port: 7860
---

# Diabetes Risk Health-Check Agent

An LLM agent that reads plain-language health details (English or Hinglish), extracts them into a strict Pydantic schema, calls a trained scikit-learn model as a tool, and explains the result. It keeps per-session memory, so you can ask "what if my BMI was 25?".

> Educational demo, not medical advice. Consult a doctor.

Built on top of an earlier ML project (Gaussian Naive Bayes + MLP on the Pima Indians diabetes data), which is kept in [`legacy/`](legacy/). Its trained models are served as version `v1`. Models are now also trained on NHANES, a public US dataset that is about 14 times larger.

## How a message flows

```
user message
   │
   ▼
FastAPI  POST /chat  (rate limit, timeout, structured logs)
   │
   ▼
1. Extraction    LLM (JSON mode) ──► Extraction schema ──► range validation (HealthInputs)
   │                                   invalid values are NOT saved; user is asked to re-check
   ▼
2. Session memory: merge new values into the saved inputs ("what if" = this turn only)
   │
   ▼
3. Agent loop (max 4 LLM calls)
   LLM ──proposes tool call──► Pydantic args check ──► "no invented values" guard ──► trained model
    ▲                                                                                   │
    └──────────────────────────── tool result (JSON) ◄──────────────────────────────────┘
   │
   ▼
4. AgentReply (structured JSON): status, risk_class, probability, model_used,
   missing_fields, invalid_fields, explanation, disclaimer  + deterministic guardrails
```

The LLM never runs the model itself. It proposes a tool call; the arguments are validated with Pydantic and checked against the values the user actually stated. Only then does our code run the pickled pipeline.

### Guardrails (in code, not just the prompt)

| Risk | What stops it |
|---|---|
| LLM invents a missing value | Tool args must match the values extracted from the user. Otherwise the call is rejected and the LLM is told to ask. |
| LLM quotes a risk number without running the model | Replies with a probability or risk level but no tool result are blocked and retried. If that fails, a fixed message is used. |
| Invalid JSON from the extractor | One retry with the parse error shown to the model, then a safe empty extraction. |
| Endless tool loop | Hard cap of 4 LLM calls per message. |
| Out-of-range values (BMI 200, glucose 7.2) | `HealthInputs` ranges; bad values are never saved or sent to the model. |
| Off-topic / medication / diagnosis requests | Classified at extraction; no tools are offered and the reply is a polite refusal. |
| High risk without advice to see a doctor | A doctor recommendation is appended if the reply lacks one. |
| Disclaimer missing | Added by the API to every reply, never left to the LLM. |

## Tools

| Tool | What it does |
|---|---|
| `predict_diabetes_risk(age, glucose, bmi, insulin?, blood_pressure?, pregnancies?, skin_thickness?, pedigree?, model="mlp")` | Picks the best model tier for the values given, applies that version's saved preprocessing, and returns class, probability, model, dataset, tier and the tier's test metrics. |
| `compare_models(...)` | Same inputs; runs MLP and Naive Bayes and reports whether they agree. |

**Minimum inputs: age, glucose and BMI.** The other five are optional. The agent never asks for them before predicting, but uses them when given. The dataset (model set) comes from the session settings, never from the LLM.

## Datasets and model tiers

| Dataset | Rows | Glucose measure | Label |
|---|---|---|---|
| **NHANES 2009–2018** (US CDC, public domain), default | 10,834 adults | Fasting plasma glucose | Doctor-diagnosed diabetes or HbA1c ≥ 6.5% (glucose is not part of the label) |
| Pima Indians (original assignment) | 768 women 21+ | 2-hour glucose tolerance test | Diabetes within 5 years |
| Kaggle "Diabetes prediction dataset", **comparison only** | 57,388 adults after cleaning (from 100,000) | Not stated | `diabetes` column, which follows an HbA1c/glucose cut-off |

**About the Kaggle set.** It's included as a comparison and is never the default:
- **Undocumented origin:** it looks synthetic. Glucose and HbA1c each take only 18 distinct values.
- **Hard-rule label:** every row with HbA1c > 6.6 or glucose > 200 is labelled diabetic.
- **HbA1c deliberately left out:** using it would let the model read the label back.
- **Cleaning:** BMI placeholder rows (27.32), children and duplicates were removed (`python -m health_agent.kaggle`).
- **Inputs:** it has no insulin, blood pressure or pregnancies, so only the minimum tier exists.

The datasets are **never merged**: a fasting 160 and a 2-hour 160 mean different things. NHANES is the default because people usually know their fasting sugar. Build it with `python -m health_agent.nhanes`, which downloads 38 CDC files (~61 MB, cached, git-ignored) and writes `data/nhanes.csv`.

Each training run (`python -m health_agent.training --dataset nhanes|pima`) trains every **tier** for both models:

| Tier | Inputs |
|---|---|
| minimum | age, glucose, BMI |
| standard | + insulin |
| extended | + blood pressure, pregnancies |
| extended_insulin | + insulin, blood pressure, pregnancies |
| full | all 8 (Pima only; NHANES has no skin-fold or pedigree data) |

- **Leak-free preprocessing:** medians and outlier bounds are fitted on the training split only.
- **Recorded metrics:** each tier records test metrics plus 5-fold cross-validated F1.
- **Tier choice:** a prediction uses the tier with the best CV F1 among those whose inputs the user gave. Extra values are only used where they helped on held-out data.
- **Upgrade hint:** if a tier that needs more values scores higher, the reply says which values to add.
- **Versioning:** every run is a new version under `models/versions/vN`. It goes live only if its F1 is not lower than the live version's on the same test rows. The decision is stored with a bootstrap 95% CI, so you can tell real differences from noise.

### Measured model results (MLP; held-out test split, 5-fold CV on the training split)

| Dataset · version | Tier | Accuracy | F1 (test) | F1 (CV) | ROC-AUC |
|---|---|---|---|---|---|
| NHANES · v3 (2,167 test rows) | minimum (age, glucose, BMI) | 91.0% | 0.715 | 0.719 ± 0.020 | 0.931 |
| NHANES · v3 | + insulin | 90.7% | 0.712 | 0.726 ± 0.014 | 0.929 |
| NHANES · v3 | + blood pressure, pregnancies | 89.6% | 0.689 | 0.707 ± 0.034 | 0.913 |
| Pima · v1 (original, live) | age, glucose, BMI, insulin | 77.9% | 0.696 | – | 0.821 |
| Pima · 8-feature retrain (experiment, not kept) | minimum | 76.0% | 0.667 | 0.583 ± 0.041 | 0.822 |
| Pima · 8-feature retrain | + blood pressure, pregnancies | 74.0% | 0.623 | 0.641 ± 0.073 | 0.801 |
| Kaggle · v4 (11,478 test rows) | minimum | 92.9% | 0.576 | 0.561 | 0.911 |

How to read this:
- **Accuracy vs. baseline:** 82% of NHANES adults are non-diabetic, so 91% accuracy should be compared with that 82%. ROC-AUC (0.93) and F1 are the more useful numbers.
- **More values helped little on NHANES:** with the larger dataset, extra values barely changed the scores, and the differences sit within CV noise. That's why the app only suggests adding values when the gain is bigger than the noise.
- **Different measures:** NHANES uses fasting glucose and Pima a 2-hour test, so their scores aren't directly comparable.
- **Kaggle scores:** its accuracy is high only because 88% of rows are non-diabetic. F1 (0.58) is the honest number: without HbA1c, glucose, age and BMI can't recover a label built mostly from HbA1c. A bigger dataset isn't a better one.

## Design

The chat and admin pages use a warm "coffee-house menu" look: an espresso header band, cream body, values listed like a café menu, and a ring gauge for risk. It's phone first; desktop puts the form beside the chat. It was chosen by comparing four full mock-ups (pinboard, report slip, coffee menu, what-if plates).

## Chat settings panel

The "Form" tab (phone) or the side panel (desktop) offers:
- **Your values:** age, sugar and BMI, plus insulin and a "More values" section for the rest. Values are sent as structured data, so they skip LLM extraction but still go through range validation.
- **Learned from:** US adults, fasting sugar (NHANES, default) · Pima women, 2-hour sugar test · Kaggle set (comparison only). It shows the size and measured accuracy in plain words.
- **Method:** Neural net, Naive Bayes, or compare both.
- **Assistant:** only LLMs with a configured key are listed, plus "Offline (no AI)".

The prompt version (always v2) and temperature (always 0) aren't exposed to users. They're developer settings, and v1 is still available to `evals/run.py --prompt-version v1`. `GET /config` serves the options.

## Admin: upload data, retrain, roll back

Set `ADMIN_TOKEN` in `.env` and open `/admin`.
- **Upload:** a CSV with `Outcome` (0/1) plus at least `Age`, `Glucose` and `BMI`. Optional columns use Pima names; missing values = 0. Rows are validated, merged into the chosen dataset with duplicates skipped (a backup is kept), and every tier is retrained in the background.
- **Go-live rule:** the new version goes live only if F1 doesn't drop, unless "activate anyway" is ticked.
- **Versions table:** every version with its metrics, and one-click roll back.

API: `POST /admin/dataset?dataset=...` (body `text/csv`), `GET /admin/jobs/{id}`, `GET /admin/models`, `POST /admin/models/{v}/activate`. All need the `X-Admin-Token` header. Admin is disabled when `ADMIN_TOKEN` is unset.

## Run it

```bash
pip install -r requirements.txt

# Offline (no API key): regex baseline instead of an LLM
LLM_PROVIDER=rules uvicorn health_agent.server:app --port 7860

# Groq (free tier) / Gemini / local Qwen via Ollama
LLM_PROVIDER=groq GROQ_API_KEY=... uvicorn health_agent.server:app --port 7860
LLM_PROVIDER=gemini GEMINI_API_KEY=... uvicorn health_agent.server:app --port 7860
LLM_PROVIDER=ollama LLM_MODEL=qwen3.5:4b uvicorn health_agent.server:app --port 7860
```

Open http://localhost:7860 for the chat page, or http://localhost:7860/docs for the API docs.

All providers go through one OpenAI-compatible client (`health_agent/llm.py`), so switching is an env var. If the primary model is overloaded (429/5xx), the client retries with backoff and then switches to a backup model (`LLM_FALLBACK_MODELS`).

### Settings (environment variables)

Set them in the shell, as host secrets, or in a local `.env` file. `.env` is git- and docker-ignored and read at startup, and real environment variables win over it.

| Variable | Example | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `groq`, `ollama`, `openai` or `rules` (offline) |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | | Key for the chosen provider |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Override the provider's default model |
| `LLM_FALLBACK_MODELS` | `gemini-3-flash-preview` | Backup models when the primary is busy (empty = none) |
| `ADMIN_TOKEN` | a long random string | Enables `/admin`; admin is off when unset |
| `RATE_LIMIT_PER_HOUR` | `20` | Chat messages per IP per hour |
| `STATS_FILE` | `/data/stats.json` | Persist the anonymous usage counts |

### Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /chat` | `{"session_id": "...?", "message": "..."}` → `{session_id, reply: AgentReply, latency_ms, tokens}` |
| `GET /health` | Model and LLM status |
| `GET /stats` | Anonymous usage counts (sessions, messages, tool calls, statuses). No health values, IPs or message text are stored. |

Example reply:

```json
{
  "status": "result",
  "risk_class": "high",
  "probability": 0.9935,
  "model_used": "mlp",
  "dataset": "nhanes",
  "model_version": "v3",
  "tier": "extended",
  "features_used": ["age", "glucose", "bmi", "blood_pressure", "pregnancies"],
  "inputs_used": {"age": 52, "glucose": 180.0, "bmi": 25.0, "blood_pressure": 90.0, "pregnancies": 2},
  "missing_fields": [],
  "what_if": true,
  "explanation": "If your BMI were 25, the model still estimates a higher risk...",
  "disclaimer": "Educational demo, not medical advice. Consult a doctor."
}
```

## Docker and deploy

```bash
docker build -t diabetes-agent .
docker run -p 7860:7860 -e LLM_PROVIDER=groq -e GROQ_API_KEY=... diabetes-agent
```

- **Hugging Face Spaces:** create a Docker Space and push this folder (the YAML header above configures it). Add `GROQ_API_KEY` and `LLM_PROVIDER=groq` as Space secrets.
- **Render:** New Web Service from the repo, runtime Docker, same env vars.
- Rate limit: `RATE_LIMIT_PER_HOUR` (default 20 per IP) protects the free LLM quota.

## Evaluation

`evals/cases.jsonl` holds 50 hand-written cases. `evals/run.py` runs them and prints a results table.

| Group | Cases | What is checked |
|---|---|---|
| Complete inputs (English) | 15 | Every stated field extracted correctly (incl. blood pressure / pregnancies in some cases), right tool called |
| Complete inputs (Hinglish) | 10 | Same, mixed Hindi-English |
| Missing fields | 10 | Missing age, glucose or BMI: asks a follow-up, lists the right missing fields, invents nothing, no tool call |
| "What if" follow-ups | 5 | Only the changed field differs, tool re-runs, saved values unchanged |
| Out-of-range values | 5 | Flagged as invalid, model not run |
| Off-topic / unsafe | 5 | Refused, no tool call |

A case passes only if every check passes. The runner also logs invented values, tool calls rejected by the guard, latency (median and p95) and tokens per conversation.

```bash
python evals/run.py --provider rules                                   # offline baseline
python evals/run.py --provider ollama --model qwen3.5:4b               # local LLM
python evals/run.py --provider gemini --prompt-version v1              # before: zero-shot prompt
python evals/run.py --provider gemini --prompt-version v2              # after: rules + few-shot
python evals/compare.py evals/results/<run_v1>.json evals/results/<run_v2>.json   # before/after table
```

Each run ends with a **Resume numbers** block. If the provider is down (503/timeout), the case is retried after 30 s rather than scored as an agent mistake; cases that still fail are listed and left out of the scores.

### Where each resume number comes from

| Number | Source |
|---|---|
| Field-extraction accuracy (overall) | `evals/run.py` → Resume numbers |
| Field-extraction accuracy (Hinglish) | same block, the 10 Hinglish cases |
| Tool-call accuracy | same block: right tool, valid arguments, called only when it should be |
| Before/after (e.g. Hinglish v1 → v2) | `evals/compare.py` on the v1 and v2 result files |
| "Used by N people" | `GET /stats` → `unique_visitors` (and `visitors_with_a_result`) on the live app |

**Counting people honestly.** The chat page makes a random id once per browser and stores it in localStorage. The server keeps only a hash of it, so `/stats` can count unique visitors without storing names, IPs or health values. To keep the count across restarts, set `STATS_FILE` to a path on persistent storage. Free Hugging Face Spaces and Render disks reset on restart, so note the number down before redeploying, or attach persistent storage.

### Results so far

<!-- RESULTS:START -->
Measured 2026-11-24 on the current 50-case set (8-field agent, age/glucose/BMI required). One run each, temperature 0.

| Configuration | Pass rate | Field accuracy (all / Hinglish) | Tool-call correct | Refusals correct | Invented values | Median / p95 latency | Tokens per conversation |
|---|---|---|---|---|---|---|---|
| Regex baseline (no LLM) | 94% | 99% / 99% | 94% | 100% | 0 | 13 ms / 26 ms | 0 |
| Gemini 3.5 Flash-Lite, prompt v1 (zero-shot) | 96% | 100% / 100% | 98% | 80% | 0 | 2.8 s / 9.7 s | 2,922 |
| **Gemini 3.5 Flash-Lite, prompt v2 (rules + few-shot)** | **100%** | **100% / 100%** | **100%** | **100%** | **0** | 3.1 s / 8.4 s | 3,879 |

What the v1 → v2 prompt change fixed (`evals/compare.py`):
- **Out-of-range values caught: 80% → 100%.** With v1, the model *converted* "sugar 7.2" (mmol/L) to 129.6 mg/dL by itself and ran the prediction. v2's rule "copy numbers exactly, don't convert units" makes validation catch it, so the user is asked to re-check instead.
- **Unsafe requests refused: 80% → 100%.** v1 didn't flag "can I stop taking my metformin?" as unsafe. v2's intent definitions do.
- The cost of v2 is about 33% more tokens per conversation (the few-shot examples).

Limits of these numbers:
- **Ceiling:** the set is small and hand-written, and v2 hits 100%, so it can't show further gains. Extraction (English and Hinglish) was already 100% with v1, so this run gives no "Hinglish X% → Y%" improvement.
- **Next step:** a harder held-out set (messier phrasing, units, corrections mid-conversation) would give a more informative number.
<!-- RESULTS:END -->

Per-case results are saved in `evals/results/`.

## Tests

```bash
pip install -r requirements.txt pytest
pytest -q tests
```

- `test_model_tool.py` checks known inputs and their expected classes, so later changes can't silently break the model.
- `test_schemas.py` checks ranges, required fields and extra-field rejection.
- `test_agent.py` uses a scripted fake LLM to test the tool loop, the invented-value guard, the step cap, blocked made-up results, JSON retry, refusals, LLM outages and what-if memory.
- `test_api.py` covers the endpoints, input validation and the rate limit.
- `test_admin.py` covers admin auth, CSV validation, merge/dedupe with backup, version listing, rollback, `/config` and chat settings. It runs on a temporary copy of the models and data.
- `test_llm.py` checks backup-model fallback on 503, retry limits, no retry on 400, and tolerant JSON parsing (mocked HTTP).

CI (`.github/workflows/ci.yml`) runs on every push:
- the tests and the offline evaluation;
- the Gemini evaluation, only when a `GEMINI_API_KEY` repository secret exists;
- a Docker build plus smoke test (build the image, start it, check `/health`, send one chat message).

## Project layout

```
.
├── health_agent/          the agent service (Python package)
│   ├── server.py          FastAPI app: /chat, /config, /health, /stats, admin API
│   ├── agent.py           extraction → validation → tool loop → structured reply
│   ├── schemas.py         Pydantic models (inputs, tool args, replies)
│   ├── prompts.py         extraction + agent prompts, tool specs
│   ├── llm.py             one client for Gemini / Groq / Ollama / OpenAI
│   ├── rules.py           offline regex baseline (no LLM)
│   ├── model_tool.py      runs the trained models (tier choice, preprocessing)
│   ├── training.py        tiers, CV, versioning, promotion rule
│   ├── nhanes.py          builds data/nhanes.csv from CDC files
│   └── kaggle.py          builds data/kaggle.csv
├── web/                   chat page and admin page (plain HTML/CSS/JS)
├── models/                registry.json + versions/vN (trained model sets)
├── data/                  training CSVs (Pima, NHANES, Kaggle); raw downloads are git-ignored
├── evals/                 cases.jsonl, run.py, compare.py, results/
├── tests/                 pytest suite
├── legacy/                the original assignment (training scripts, Flask app, plots); see legacy/README.md
├── Dockerfile
└── requirements.txt       dependencies
```

## Scaling notes

Sessions, rate limits and stats live in memory, so this is a single-instance demo. To scale: keep the API stateless, put sessions and rate limits in Redis, and put LLM calls behind a queue with per-provider concurrency limits.

## Author

Aryan Agarwal
