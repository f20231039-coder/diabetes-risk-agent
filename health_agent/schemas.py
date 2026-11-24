"""Pydantic v2 schemas: extraction output, validated inputs, tool args, replies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

REQUIRED_FOR_PREDICTION = ("age", "glucose", "bmi")  # minimum the models need
OPTIONAL_FIELDS = ("insulin", "blood_pressure", "pregnancies", "skin_thickness", "pedigree")
FIELDS = REQUIRED_FOR_PREDICTION + OPTIONAL_FIELDS

# Valid ranges. Anything outside is rejected and the user is asked to re-check.
RANGES = {
    "age": (1, 120),
    "glucose": (40, 400),  # mg/dL
    "bmi": (10, 80),
    "insulin": (0, 900),  # uU/mL
    "blood_pressure": (30, 150),  # diastolic, mm Hg
    "pregnancies": (0, 20),
    "skin_thickness": (5, 100),  # triceps skin fold, mm
    "pedigree": (0.05, 3.0),  # diabetes pedigree function
}

DISCLAIMER = "Educational demo, not medical advice. Consult a doctor."

Intent = Literal["health_info", "what_if", "compare", "question", "greeting", "off_topic", "unsafe"]
ModelChoice = Literal["mlp", "naive_bayes"]


class Extraction(BaseModel):
    """What the LLM extracts from ONE user message. No range checks here, so the
    model reports exactly what the user said and validation happens afterwards."""

    model_config = ConfigDict(extra="ignore")

    age: float | None = None
    glucose: float | None = None
    bmi: float | None = None
    insulin: float | None = None
    blood_pressure: float | None = None
    pregnancies: float | None = None
    skin_thickness: float | None = None
    pedigree: float | None = None
    intent: Intent = "health_info"
    model_preference: ModelChoice | None = None

    def stated(self) -> dict[str, float]:
        return {f: getattr(self, f) for f in FIELDS if getattr(self, f) is not None}


def _range(name: str):
    lo, hi = RANGES[name]
    return Field(None, ge=lo, le=hi)


class HealthInputs(BaseModel):
    """All model inputs; each optional, each range-checked."""

    model_config = ConfigDict(extra="forbid")

    age: int | None = _range("age")
    glucose: float | None = _range("glucose")
    bmi: float | None = _range("bmi")
    insulin: float | None = _range("insulin")
    blood_pressure: float | None = _range("blood_pressure")
    pregnancies: int | None = _range("pregnancies")
    skin_thickness: float | None = _range("skin_thickness")
    pedigree: float | None = _range("pedigree")

    def missing(self) -> list[str]:
        """Required fields still unknown (the minimum to run a prediction)."""
        return [f for f in REQUIRED_FOR_PREDICTION if getattr(self, f) is None]

    def known(self) -> dict:
        return {f: v for f, v in self.model_dump().items() if v is not None}


def validate_fields(values: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
    """Validate each field independently. Returns (valid_values, errors_by_field)."""
    valid: dict[str, float] = {}
    errors: dict[str, str] = {}
    for name, value in values.items():
        if name not in RANGES:
            continue
        try:
            parsed = HealthInputs.model_validate({name: value})
            valid[name] = getattr(parsed, name)
        except ValidationError:
            lo, hi = RANGES[name]
            errors[name] = f"{name}={value:g} is outside the accepted range {lo}-{hi}"
    return valid, errors


class _ArgsBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=1, le=120, description="Age in years")
    glucose: float = Field(..., ge=40, le=400, description="Glucose in mg/dL")
    bmi: float = Field(..., ge=10, le=80, description="Body mass index")
    insulin: float | None = Field(None, ge=0, le=900)
    blood_pressure: float | None = Field(None, ge=30, le=150)
    pregnancies: int | None = Field(None, ge=0, le=20)
    skin_thickness: float | None = Field(None, ge=5, le=100)
    pedigree: float | None = Field(None, ge=0.05, le=3.0)


class PredictArgs(_ArgsBase):
    """Arguments the LLM must supply to predict_diabetes_risk."""

    model: ModelChoice = Field("mlp", description="Which trained model to use")


class CompareArgs(_ArgsBase):
    """Arguments for compare_models (runs both models)."""


Status = Literal["result", "need_info", "invalid_input", "refused", "chat", "error"]


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict
    ok: bool
    error: str | None = None


class AgentReply(BaseModel):
    """The structured reply returned by /chat."""

    status: Status
    risk_class: Literal["low", "high"] | None = None
    probability: float | None = None
    model_used: str | None = None
    model_version: str | None = None
    dataset: str | None = None
    tier: str | None = None
    features_used: list[str] = []
    tier_test_metrics: dict | None = None
    more_values_could_help: dict | None = None
    comparison: dict | None = None
    inputs_used: HealthInputs | None = None
    known_inputs: HealthInputs
    missing_fields: list[str] = []
    invalid_fields: list[str] = []
    insulin_imputed: bool = False
    what_if: bool = False
    explanation: str
    disclaimer: str = DISCLAIMER
    tool_calls: list[ToolCallRecord] = []
