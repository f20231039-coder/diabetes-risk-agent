# Diabetes Risk Health-Check Agent

A chat assistant that estimates diabetes risk. You describe your age, blood sugar and BMI in plain English or Hinglish (for example *"meri age 45 hai, fasting sugar 160 aayi, BMI 29"*). An LLM pulls the values into a strict schema, calls a trained machine-learning model as a tool, and explains the result. It remembers the conversation, so you can ask *"what if my BMI was 25?"*.

**Live demo:** https://diabetes-risk-agent.onrender.com (free hosting: the first visit after a quiet period takes about a minute to wake up)

> Educational demo, not medical advice. Consult a doctor.

**Stack:** Python, FastAPI, Pydantic, scikit-learn, Gemini (or Groq / Ollama), plain HTML/CSS/JS, Docker, GitHub Actions.

## How it works

```
message ─► 1. Extract   LLM fills a Pydantic schema; values outside valid ranges are rejected
        ─► 2. Remember  new values are merged into the session ("what if" changes are temporary)
        ─► 3. Act       LLM proposes a tool call ─► arguments are validated ─► trained model runs
        ─► 4. Reply     structured JSON (risk, probability, model, inputs used) + plain explanation
```

The LLM never runs the model or invents numbers. Guardrails enforced in code:

- A tool call is rejected if any value differs from what the user actually said.
- A reply that states a risk or probability without a model result is blocked.
- Invalid JSON gets one retry. The tool loop is capped at 4 steps.
- Off-topic, diagnosis and medication questions are refused, and no tools are offered for them.
- Every reply carries the disclaimer. High-risk replies always recommend seeing a doctor.

Only **age, glucose and BMI** are required. Insulin, blood pressure, pregnancies, skin-fold thickness and family-history score are optional. When given, they let the agent use a model trained on more features.

## Results

### Agent evaluation

The set is 50 hand-written cases: 15 English, 10 Hinglish, 10 with missing values, 5 "what if", 5 out-of-range and 5 off-topic or unsafe. A case passes only if every check passes.

| Configuration | Pass rate | Field extraction (all / Hinglish) | Correct tool calls | Correct refusals | Invented values |
|---|---|---|---|---|---|
| Regex baseline, no LLM | 94% | 99% / 99% | 94% | 100% | 0 |
| Gemini 3.5 Flash-Lite, prompt v1 (basic) | 96% | 100% / 100% | 98% | 80% | 0 |
| **Gemini 3.5 Flash-Lite, prompt v2 (rules + examples)** | **100%** | **100% / 100%** | **100%** | **100%** | **0** |

Prompt v2 fixed two failures of v1:
- **Unit conversion:** v1 silently converted "sugar 7.2" (mmol/L) to mg/dL instead of asking the user to re-check.
- **Missed refusal:** v1 answered "can I stop taking my metformin?" instead of refusing.

The set is small and v2 reaches 100%, so a harder held-out set is the next step.

### Models

| Dataset | People | Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **NHANES 2009–2018** (US adults, fasting glucose), default | 10,834 | MLP, age + glucose + BMI | 91.0% | 0.72 | 0.93 |
| Pima Indians (women 21+, 2-hour glucose test) | 768 | MLP, + insulin (original model) | 77.9% | 0.70 | 0.82 |
| Kaggle diabetes prediction, comparison only | 57,388 | MLP, age + glucose + BMI | 92.9% | 0.58 | 0.91 |

Notes on the models:
- **Held-out data:** every score comes from a test set the model never saw. Preprocessing is fitted on training data only, and each version also records 5-fold cross-validation.
- **Read accuracy against the base rate:** 82% of NHANES adults are non-diabetic, so ROC-AUC and F1 say more than accuracy.
- **Never merged:** the datasets measure glucose differently (fasting vs 2-hour test).
- **Why Kaggle is comparison only:** the data looks synthetic, and its label is derived from HbA1c, which is deliberately not used as an input. That's why its F1 is low despite its size.

## Run locally

```bash
pip install -r requirements.txt
LLM_PROVIDER=gemini GEMINI_API_KEY=your-key uvicorn health_agent.server:app --port 7860
```

Open http://localhost:7860. API docs are at `/docs`. Without a key, use `LLM_PROVIDER=rules` for the offline baseline.

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini`, `groq`, `ollama`, `openai` or `rules` |
| `GEMINI_API_KEY` / `GROQ_API_KEY` | API key for the chosen provider |
| `LLM_MODEL` | Override the default model, e.g. `gemini-3.5-flash-lite` |
| `ADMIN_TOKEN` | Enables the `/admin` page (off when unset) |
| `RATE_LIMIT_PER_HOUR` | Messages per IP per hour (default 20) |

Variables can also go in a local `.env` file, which is git-ignored.

## Features

- **Chat page:** a phone-first chat with a result card (risk gauge, inputs used, how it was worked out), "what if" questions and model comparison. A form tab lets you enter values directly and choose the dataset and method (neural net, Naive Bayes, or both).
- **Admin page** (`/admin`): upload a CSV of new rows, retrain all models in the background, and compare versions. A new version goes live only if its F1 doesn't drop, and you can roll back with one click.
- **Endpoints:** `POST /chat`, `GET /config`, `GET /health`, and `GET /stats` (anonymous counts only: no health values, IPs or messages are stored).

## Development

```bash
pip install -r requirements.txt pytest
pytest -q tests                                   # 50 tests
python evals/run.py --provider rules              # evaluation, offline
python evals/run.py --provider gemini             # evaluation with the LLM
python -m health_agent.training --dataset nhanes  # retrain a model set
```

On every push, CI runs the tests, the offline evaluation, and a Docker build with a smoke test.

## Deploy (free, on Render)

1. Sign in at [render.com](https://render.com) with GitHub, then choose **New → Web Service** and pick this repository.
2. Use these settings:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn health_agent.server:app --host 0.0.0.0 --port $PORT`
   - **Instance type:** Free
   - **Health check path:** `/health`
3. Add environment variables: `GEMINI_API_KEY` (your key), `LLM_PROVIDER=gemini`, `LLM_MODEL=gemini-3.5-flash-lite`, and optionally `ADMIN_TOKEN`.

Free services sleep when idle, so the first visit after a while takes about a minute to wake up. Sessions and stats are kept in memory, so they reset on restart. The `Dockerfile` runs the same app on any Docker host: `docker build -t diabetes-agent . && docker run -p 7860:7860 -e GEMINI_API_KEY=... diabetes-agent`.

## Project structure

```
health_agent/   the service: API (server.py), agent loop, schemas, prompts, LLM client,
                model runner, training and dataset builders
web/            chat and admin pages
models/         trained model versions and the registry of live versions
data/           training data (Pima, NHANES, Kaggle)
evals/          evaluation cases and runner
tests/          test suite
legacy/         the original Naive Bayes / MLP project this grew from (see legacy/README.md)
```

## Author

Aryan Agarwal
