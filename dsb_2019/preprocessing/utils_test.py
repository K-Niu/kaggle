import pandas as pd
import pytest

from dsb_2019.preprocessing.utils import (
    clean_column_name,
    subsample_assessments,
    regression_scores_to_pred,
)


def test_clean_column_name():
    assert (
        clean_column_name("Welcome to Lost Lagoon!_2000")
        == "Welcome_to_Lost_Lagoon_2000"
    )
    assert (
        clean_column_name("Sandcastle Builder (Activity)_2000")
        == "Sandcastle_Builder_Activity_2000"
    )
    assert (
        clean_column_name("Heavy, Heavier, Heaviest_2000_count")
        == "Heavy_Heavier_Heaviest_2000_count"
    )


def test_subsample_assessments():
    data_df = pd.DataFrame(
        [
            {"installation_id": "a", "assessment": "Mushroom Sorter (Assessment)"},
            {"installation_id": "a", "assessment": "Bird Measurer (Assessment)"},
            {"installation_id": "b", "assessment": "Mushroom Sorter (Assessment)"},
            {"installation_id": "b", "assessment": "Bird Measurer (Assessment)"},
        ]
    )

    assert len(subsample_assessments(data_df)) == len(
        data_df["installation_id"].unique()
    )


def test_regression_scores_to_pred():
    scores = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    assert regression_scores_to_pred([0.2, 0.2, 0.2, 0.4], scores) == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        3,
        3,
    ]
    with pytest.raises(AssertionError):
        regression_scores_to_pred([0.1, 0.1, 0.1, 0.1], scores)
