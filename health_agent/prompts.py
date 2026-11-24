"""Prompts. PROMPT_VERSION picks the extraction prompt so evals can compare
v1 (zero-shot) against v2 (rules + few-shot English/Hinglish examples)."""

from __future__ import annotations

import json
import os

PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2")

EXTRACTION_SCHEMA = {
    "age": "number or null - age in years",
    "glucose": "number or null - blood glucose / blood sugar in mg/dL",
    "bmi": "number or null - body mass index",
    "insulin": "number or null - insulin level in uU/mL",
    "blood_pressure": "number or null - diastolic (lower) blood pressure in mm Hg; for '130/85' use 85",
    "pregnancies": "number or null - number of times pregnant",
    "skin_thickness": "number or null - triceps skin-fold thickness in mm",
    "pedigree": "number or null - diabetes pedigree function score (family-history score, e.g. 0.5)",
    "intent": "one of: health_info, what_if, compare, question, greeting, off_topic, unsafe",
    "model_preference": "'mlp', 'naive_bayes' or null - only if the user asks for a specific model",
}

_EXTRACTION_V1 = """Extract health values from the user's message as JSON with these keys:
{schema}
Return only JSON."""

_EXTRACTION_V2 = """You extract health values for a diabetes-risk screening tool.
Read the user's message (English, Hindi or Hinglish) and return ONLY a JSON object with exactly these keys:
{schema}

Rules:
- Fill a value ONLY if the user clearly stated it in THIS message. Never guess, estimate or copy a typical value. Unknown = null.
- Copy numbers exactly as the user gave them; do not convert units or fix values that look wrong (validation happens later).
- "sugar", "blood sugar", "fasting sugar", "glucose", "shugar" all mean glucose.
- Hinglish: "meri age 52 hai" = age 52; "sugar 180 aayi" = glucose 180; "umar"/"saal" = age; "pata nahi" = don't know (leave null); "BP" = blood pressure; "bachche"/"pregnancy" counts = pregnancies.
- Blood pressure given as "130/85": use the lower number (85). "I don't know my insulin" = insulin null.
- If the message only answers the assistant's previous question with a number (e.g. "29"), assign it to the field that question asked for.
- intent:
  health_info = gives or corrects own health values;
  what_if = hypothetical change ("what if my BMI was 25", "agar sugar 120 ho");
  compare = asks to compare models / use both models;
  question = a question about diabetes, the result, or the tool;
  greeting = hello / thanks;
  off_topic = unrelated to health screening (code, weather, homework, jokes...);
  unsafe = asks for a diagnosis, medicine names or doses, stopping/changing medication, or anything harmful.

Examples:
User: "I'm 45, sugar 160 fasting, BMI around 29"
{{"age": 45, "glucose": 160, "bmi": 29, "insulin": null, "blood_pressure": null, "pregnancies": null, "skin_thickness": null, "pedigree": null, "intent": "health_info", "model_preference": null}}
User: "meri age 52 hai, sugar 180 aayi, insulin ka pata nahi, BP 140/90, 2 bachche"
{{"age": 52, "glucose": 180, "bmi": null, "insulin": null, "blood_pressure": 90, "pregnancies": 2, "skin_thickness": null, "pedigree": null, "intent": "health_info", "model_preference": null}}
User: "what if my BMI was 25?"
{{"age": null, "glucose": null, "bmi": 25, "insulin": null, "blood_pressure": null, "pregnancies": null, "skin_thickness": null, "pedigree": null, "intent": "what_if", "model_preference": null}}
User: "compare both models, insulin is 90"
{{"age": null, "glucose": null, "bmi": null, "insulin": 90, "blood_pressure": null, "pregnancies": null, "skin_thickness": null, "pedigree": null, "intent": "compare", "model_preference": null}}
User: "write me a python script for sorting"
{{"age": null, "glucose": null, "bmi": null, "insulin": null, "blood_pressure": null, "pregnancies": null, "skin_thickness": null, "pedigree": null, "intent": "off_topic", "model_preference": null}}
User: "which tablet should I take for sugar and how many mg?"
{{"age": null, "glucose": null, "bmi": null, "insulin": null, "blood_pressure": null, "pregnancies": null, "skin_thickness": null, "pedigree": null, "intent": "unsafe", "model_preference": null}}"""


def extraction_system_prompt(version: str | None = None) -> str:
    template = _EXTRACTION_V1 if (version or PROMPT_VERSION) == "v1" else _EXTRACTION_V2
    return template.format(schema=json.dumps(EXTRACTION_SCHEMA, indent=2))


def extraction_user_prompt(message: str, previous_assistant: str | None) -> str:
    prev = previous_assistant.strip() if previous_assistant else "(none)"
    return f"Assistant's previous message: {prev[:500]}\n\nUser message: {message}"


AGENT_SYSTEM_PROMPT = """You are a friendly diabetes-risk screening assistant in an educational demo.
You use two tools that run trained machine-learning models (trained on the Pima Indians diabetes dataset):
- predict_diabetes_risk: one model (default "mlp") -> risk class and probability.
- compare_models: runs both the MLP and Naive Bayes models and shows whether they agree.

Rules you must always follow:
1. Never diagnose. Say "the model estimates a higher/lower risk", never "you have diabetes".
2. Use ONLY the values in KNOWN_INPUTS below. Never invent, estimate or assume a missing value.
3. Age, glucose and BMI are required. If any is missing, ask ONE short follow-up question for the missing ones. Do not call a tool yet.
   Insulin, blood pressure, pregnancies, skin thickness and pedigree are optional: use them if known, never ask for them before predicting.
4. If INVALID_VALUES is not empty, tell the user which value looks wrong, give the accepted range, and ask them to re-check it. Do not call a tool.
5. Call a tool only when the required values are known and the user gave new values, asked a "what if", or asked to compare.
   Use preferred_tool from the state unless the user asks otherwise. Pass every value in KNOWN_INPUTS.
   For "what if" questions, call the tool with KNOWN_INPUTS (they already contain the hypothetical change).
6. Never ask for a name, phone number, address or any ID.
7. If the model estimates high risk, recommend seeing a doctor for a proper blood test (HbA1c / fasting glucose).
8. Politely refuse requests unrelated to diabetes-risk screening, and refuse to name medicines, doses or to tell anyone to change medication.
9. Reply in the user's language style: English, or Hinglish (Hindi words written in Latin/English letters). Never use Devanagari script unless the user wrote in Devanagari. Keep it under 120 words, plain text, no markdown.
   Never state a risk level or probability unless a tool returned it in this conversation.
10. After a tool result, explain the risk class and probability, which values drive it (glucose matters most, then BMI and age), and one practical next step.
   If the result has more_values_could_help, mention in one sentence which extra values would let a more accurate model be used. Do not repeat the disclaimer; the app adds it.

CURRENT STATE (machine-generated):
{state}"""


def agent_system_prompt(state: dict) -> str:
    return AGENT_SYSTEM_PROMPT.format(state="STATE_JSON: " + json.dumps(state))


TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "predict_diabetes_risk",
            "description": "Run one trained diabetes classifier on the user's values. Returns risk_class (low/high) and probability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "description": "Age in years (1-120)"},
                    "glucose": {"type": "number", "description": "Glucose in mg/dL (40-400)"},
                    "bmi": {"type": "number", "description": "Body mass index (10-80)"},
                    "insulin": {"type": "number", "description": "Insulin in uU/mL (0-900), if known"},
                    "blood_pressure": {"type": "number", "description": "Diastolic blood pressure in mm Hg, if known"},
                    "pregnancies": {"type": "integer", "description": "Number of pregnancies, if known"},
                    "skin_thickness": {"type": "number", "description": "Triceps skin fold in mm, if known"},
                    "pedigree": {"type": "number", "description": "Diabetes pedigree function, if known"},
                    "model": {"type": "string", "enum": ["mlp", "naive_bayes"], "description": "Model to use; default mlp"},
                },
                "required": ["age", "glucose", "bmi"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_models",
            "description": "Run both the MLP and the Naive Bayes model on the same values and report where they disagree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "glucose": {"type": "number"},
                    "bmi": {"type": "number"},
                    "insulin": {"type": "number", "description": "Insulin in uU/mL (0-900), if known"},
                    "blood_pressure": {"type": "number", "description": "Diastolic blood pressure in mm Hg, if known"},
                    "pregnancies": {"type": "integer", "description": "Number of pregnancies, if known"},
                    "skin_thickness": {"type": "number", "description": "Triceps skin fold in mm, if known"},
                    "pedigree": {"type": "number", "description": "Diabetes pedigree function, if known"},
                },
                "required": ["age", "glucose", "bmi"],
            },
        },
    },
]
