"""Offline regex baseline that speaks the same LLMClient interface.

It lets the app, tests and CI run without an API key, and gives the evaluation
a non-LLM baseline to compare the real LLM against.
"""

from __future__ import annotations

import json
import re

from .llm import LLMResult, ToolCall

NUM = r"(\d{1,4}(?:\.\d+)?)"

PATTERNS = {
    "age": [
        rf"\b(?:age|umar|umr|aayu)\b\D{{0,15}}?{NUM}",
        rf"{NUM}\s*(?:years?|yrs?|y/?o|saal|sal|varsh)\b",
        rf"\bi'?\s?a?m\s+{NUM}\b(?!\s*(?:kg|cm|mg|%))",
    ],
    "glucose": [
        rf"\b(?:glucose|sugar|shugar|blood sugar|fbs|rbs)\b\D{{0,20}}?{NUM}",
        rf"{NUM}\s*mg\s*/?\s*dl",
    ],
    "bmi": [rf"\bbmi\b\D{{0,15}}?{NUM}"],
    "insulin": [rf"\binsulin\b\D{{0,15}}?{NUM}"],
    "blood_pressure": [
        rf"\b(?:bp|blood pressure)\b\D{{0,15}}?\d{{2,3}}\s*/\s*{NUM}",
        rf"\b(?:bp|blood pressure|diastolic)\b\D{{0,15}}?{NUM}",
    ],
    "pregnancies": [rf"{NUM}\s*(?:pregnanc\w*|bachch?e|kids|children)", rf"\bpregnanc\w*\D{{0,10}}?{NUM}"],
    "skin_thickness": [rf"\bskin\w*(?:\s*fold)?(?:\s*thickness)?\D{{0,10}}?{NUM}"],
    "pedigree": [rf"\bpedigree\D{{0,15}}?{NUM}"],
}
FIELD_WORDS = {
    "age": r"\b(age|umar|old)\b",
    "glucose": r"\b(glucose|sugar)\b",
    "bmi": r"\bbmi\b",
    "insulin": r"\binsulin\b",
}
INSULIN_UNKNOWN = r"insulin.{0,40}(don'?t know|dont know|not sure|no idea|pata nahi|nahi pata|never (?:tested|checked)|unknown)|(?:don'?t know|pata nahi|nahi pata|no idea).{0,20}insulin"
WHAT_IF = r"\b(what if|what about if|suppose|agar|if (?:my|i)|hypothetically)\b"
COMPARE = r"\b(compare|both models|dono model|naive bayes vs|disagree)\b"
UNSAFE = r"\b(diagnos\w*|prescri\w*|dose|dosage|mg of|tablet|medicine|medication|metformin|stop taking|kill|overdose|dawai|dawa)\b"
GREETING = r"^\s*(hi|hello|hey|namaste|thanks|thank you|ok|okay|shukriya)\b[\s!.]*$"
HEALTH = r"\b(diabet\w*|sugar|glucose|insulin|bmi|risk|health|model|result|probability|weight|exercise|diet|test|hba1c)\b"


def extract(message: str, previous_assistant: str | None = None) -> dict:
    text = message.lower()
    out: dict = {k: None for k in PATTERNS}
    for fld, patterns in PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                out[fld] = float(m.group(1))
                break
    # A bare number answering the previous question ("29").
    bare = re.fullmatch(rf"\s*(?:it'?s|its|about|around|approx\.?)?\s*{NUM}\s*\.?\s*", text)
    if bare and previous_assistant:
        prev = previous_assistant.lower()
        asked = [f for f, w in FIELD_WORDS.items() if re.search(w, prev)]
        if len(asked) == 1:
            out[asked[0]] = float(bare.group(1))
    out["insulin_unknown"] = bool(re.search(INSULIN_UNKNOWN, text))
    if out["insulin_unknown"]:
        out["insulin"] = None
    stated = any(out[f] is not None for f in PATTERNS)
    if re.search(UNSAFE, text):
        intent = "unsafe"
    elif re.search(COMPARE, text):
        intent = "compare"
    elif stated and re.search(WHAT_IF, text):
        intent = "what_if"
    elif stated or out["insulin_unknown"]:
        intent = "health_info"
    elif re.search(GREETING, text):
        intent = "greeting"
    elif re.search(HEALTH, text):
        intent = "question"
    else:
        intent = "off_topic"
    out["intent"] = intent
    out["model_preference"] = "naive_bayes" if re.search(r"naive\s*bayes|\bnb\b", text) and intent != "compare" else None
    return out


def _state(messages: list[dict]) -> dict:
    for m in messages:
        if m["role"] == "system" and "STATE_JSON:" in (m.get("content") or ""):
            return json.loads(m["content"].split("STATE_JSON:", 1)[1].strip())
    return {}


def _explain(results: list[dict]) -> str:
    parts = []
    for r in results:
        if "results" in r:  # compare_models
            m, nb = r["results"]["mlp"], r["results"]["naive_bayes"]
            agree = "agree" if r["agree"] else "disagree"
            parts.append(
                f"The two models {agree}: MLP estimates {m['risk_class']} risk ({m['probability']:.0%}) and "
                f"Naive Bayes estimates {nb['risk_class']} risk ({nb['probability']:.0%})."
            )
            high = m["risk_class"] == "high" or nb["risk_class"] == "high"
        elif "risk_class" in r:
            parts.append(f"The {r['model_label']} estimates a {r['risk_class']} risk of diabetes (probability {r['probability']:.0%}).")
            high = r["risk_class"] == "high"
        else:
            continue
        hint = (r.get("more_values_could_help") if "risk_class" in r else None)
        if hint:
            parts.append(f"Adding your {', '.join(hint['ask_for'])} would let a more accurate model be used.")
        parts.append(
            "Glucose has the biggest effect, followed by BMI and age."
            + (" Please see a doctor for a proper blood test (HbA1c or fasting glucose)." if high else " Keep up regular check-ups and an active lifestyle.")
        )
    return " ".join(parts) or "I could not run the model."


class RuleBasedClient:
    name = "rules:regex-baseline"

    async def chat(self, messages, tools=None, json_mode=False) -> LLMResult:
        if json_mode:
            user = messages[-1]["content"]
            prev = re.search(r"Assistant's previous message: (.*?)\n\nUser message: ", user, re.DOTALL)
            msg = user.split("User message: ", 1)[-1]
            data = extract(msg, prev.group(1) if prev else None)
            return LLMResult(content=json.dumps(data))

        if messages[-1]["role"] == "tool":
            results = []
            for m in reversed(messages):
                if m["role"] != "tool":
                    break
                try:
                    results.insert(0, json.loads(m["content"]))
                except json.JSONDecodeError:
                    pass
            return LLMResult(content=_explain(results))

        s = _state(messages)
        intent = s.get("intent")
        if intent in ("off_topic", "unsafe"):
            return LLMResult(content="Sorry, I can only help estimate diabetes risk from age, glucose, BMI and insulin. I can't diagnose or suggest medicines; please ask a doctor.")
        if s.get("invalid_values"):
            bad = "; ".join(s["invalid_values"].values())
            return LLMResult(content=f"That doesn't look right: {bad}. Could you please re-check it?")
        if s.get("missing_fields"):
            missing = s["missing_fields"]
            return LLMResult(content=f"Could you tell me your {', '.join(missing)}? (Insulin and blood pressure are optional but can help.)")
        should_run = s.get("new_values_this_turn") or intent in ("what_if", "compare")
        if tools and should_run:
            args = dict(s["known_inputs"])
            name = "compare_models" if intent == "compare" or s.get("preferred_tool") == "compare_models" else "predict_diabetes_risk"
            if name == "predict_diabetes_risk" and s.get("model_preference"):
                args["model"] = s["model_preference"]
            return LLMResult(content=None, tool_calls=[ToolCall(id="call_0", name=name, arguments=json.dumps(args))])
        if intent == "greeting":
            return LLMResult(content="Hello! Tell me at least your age, glucose (sugar) and BMI, and I'll estimate your diabetes risk.")
        return LLMResult(content="I estimate diabetes risk from age, glucose, BMI and insulin. Share your values, or ask a 'what if' question.")
