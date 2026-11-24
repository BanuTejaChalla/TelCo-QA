"""Basic edge-case tests for Metric 04. Run: pytest -q test_metric_04_annotation_availability.py"""
import pandas as pd
from metric_04_annotation_availability import evaluate_annotation_availability

def test_fully_labeled():
    result = evaluate_annotation_availability(pd.DataFrame({"text": ["a", "b"], "sentiment": ["pos", "neg"]}))
    assert result["annotation_availability_percent"] == 100.0
    assert result["overall_score_inclusion"] is True

def test_partially_labeled():
    result = evaluate_annotation_availability(pd.DataFrame({"text": ["a", "b", "c"], "intent": ["buy", None, "help"]}))
    assert result["annotation_availability_percent"] == 66.67

def test_unlabeled_is_zero_but_excluded_from_aggregate():
    result = evaluate_annotation_availability(pd.DataFrame({"text": ["a", "b"], "source": ["x", "y"]}))
    assert result["annotation_availability_percent"] == 0.0
    assert result["status"] == "not_applicable"
    assert result["overall_score_inclusion"] is False

def test_empty_string_is_missing():
    result = evaluate_annotation_availability(pd.DataFrame({"text": ["a", "b"], "label": ["ok", " "]}))
    assert result["annotation_availability_percent"] == 50.0

def test_explicit_columns_override_detection():
    result = evaluate_annotation_availability(
        pd.DataFrame({"text": ["a"], "custom_target": ["x"]}),
        annotation_columns=["custom_target"],
    )
    assert result["annotation_columns"] == ["custom_target"]


def test_detected_but_entirely_empty_annotations_are_not_applicable():
    result = evaluate_annotation_availability(pd.DataFrame({"text": ["a", "b"], "labels": [None, ""]}))
    assert result["annotation_availability_percent"] == 0.0
    assert result["status"] == "not_applicable"
    assert result["overall_score_inclusion"] is False
