import pytest
from pydantic import ValidationError

from health_agent.schemas import Extraction, HealthInputs, PredictArgs, validate_fields


def test_health_inputs_all_optional_and_missing():
    h = HealthInputs(age=45, glucose=160)
    assert h.missing() == ["bmi"]  # only age, glucose, BMI are required
    assert HealthInputs(age=45, glucose=160, bmi=29).missing() == []
    assert HealthInputs(age=45, blood_pressure=85, pregnancies=2).known() == {"age": 45, "blood_pressure": 85, "pregnancies": 2}


@pytest.mark.parametrize("field,value", [("age", 0), ("age", 150), ("bmi", 5), ("bmi", 95), ("glucose", 7.2), ("insulin", 5000)])
def test_out_of_range_rejected(field, value):
    with pytest.raises(ValidationError):
        HealthInputs(**{field: value})


def test_validate_fields_splits_good_and_bad():
    valid, errors = validate_fields({"age": 45, "bmi": 200})
    assert valid == {"age": 45}
    assert set(errors) == {"bmi"}


def test_predict_args_forbid_extra_and_require_core_fields():
    with pytest.raises(ValidationError):
        PredictArgs(age=40, glucose=120)  # bmi missing
    with pytest.raises(ValidationError):
        PredictArgs(age=40, glucose=120, bmi=25, name="Ravi")  # no extra fields
    assert PredictArgs(age=40, glucose=120, bmi=25).model == "mlp"


def test_extraction_ignores_unknown_keys():
    e = Extraction.model_validate({"age": 52, "glucose": 180, "foo": 1})
    assert e.stated() == {"age": 52, "glucose": 180}
