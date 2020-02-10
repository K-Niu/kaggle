import pytest
import pandas as pd

from utils.feature_selection import (
    remove_correlated_features,
    remove_high_entropy_features,
)


@pytest.fixture()
def features_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"a": 1.0, "b": 2.0, "c": 1.0},
            {"a": 2.0, "b": 3.0, "c": 2.0},
            {"a": 3.0, "b": 7.0, "c": 3.0},
        ]
    )


def test_remove_correlated_features(features_df):
    assert (
        len(remove_correlated_features(features_df, feature_names=["a", "b", "c"])) == 1
    )


def test_remove_high_entropy_features(features_df):
    assert (
        len(remove_high_entropy_features(features_df, feature_names=["a", "b", "c"]))
        == 2
    )
