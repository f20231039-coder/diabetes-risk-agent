"""Guards the trained models: later changes must not silently change predictions."""

import numpy as np
import pytest

from health_agent.model_tool import build_features, compare_models, predict_diabetes_risk

# (age, glucose, bmi, insulin) -> expected class for both models. Rows 1-3 are
# from diabetes.csv (Outcome 1, 0, 1); insulin 0 in the CSV means "unknown".
KNOWN_CASES = [
    ((50, 148, 33.6, None), 1),
    ((31, 85, 26.6, None), 0),
    ((32, 183, 23.3, None), 1),
]


@pytest.mark.parametrize("model", ["mlp", "naive_bayes"])
@pytest.mark.parametrize("args,expected", KNOWN_CASES)
def test_known_inputs(model, args, expected):
    age, glucose, bmi, insulin = args
    result = predict_diabetes_risk(age=age, glucose=glucose, bmi=bmi, insulin=insulin, model=model, dataset="pima")
    assert result["prediction"] == expected
    assert result["risk_class"] == ("high" if expected else "low")
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_used"] == model


def test_missing_insulin_equals_training_zero_handling():
    # Training replaced Insulin == 0 with the median, so None and 0 must match.
    assert np.array_equal(build_features(50, 148, 33.6, None), build_features(50, 148, 33.6, 0))


def test_feature_order_and_interactions():
    x = build_features(age=30, glucose=100, bmi=25, insulin=120)[0]
    assert list(x) == [100, 120, 25, 30, 25 * 30, 100 * 120]


def test_compare_models_reports_agreement():
    out = compare_models(age=50, glucose=148, bmi=33.6, dataset="pima")
    assert set(out["results"]) == {"mlp", "naive_bayes"}
    assert out["agree"] is True
    assert out["insulin_imputed"] is True


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        predict_diabetes_risk(age=30, glucose=100, bmi=25, model="svm", dataset="pima")


@pytest.mark.parametrize("model", ["mlp", "naive_bayes"])
def test_nhanes_sanity(model):
    # Fasting glucose 220 with obesity -> high; healthy fasting values -> low.
    high = predict_diabetes_risk(age=60, glucose=220, bmi=36, model=model, dataset="nhanes")
    low = predict_diabetes_risk(age=30, glucose=88, bmi=22, model=model, dataset="nhanes")
    assert high["risk_class"] == "high" and low["risk_class"] == "low"
    assert high["dataset"] == "nhanes" and high["tier"] == "minimum"


def test_kaggle_set_is_separate_and_minimum_only():
    from health_agent import training

    if training.active_version("kaggle") is None:
        pytest.skip("kaggle model set not trained")
    r = predict_diabetes_risk(age=60, glucose=220, bmi=36, insulin=120, dataset="kaggle")
    assert r["dataset"] == "kaggle" and r["tier"] == "minimum"  # no insulin tier exists for this data
    assert training.DEFAULT_DATASET == "nhanes"


def test_tier_uses_extra_values_when_given():
    r = predict_diabetes_risk(age=45, glucose=110, bmi=29, insulin=12, dataset="nhanes")
    assert r["tier"] in ("standard", "minimum") and "age" in r["features_used"]
