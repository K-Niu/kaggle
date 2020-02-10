from typing import List
import string

import pandas as pd
import numpy as np
import pytest


def clean_column_name(name: str) -> str:
    """
    Strips non-essential characters from column name
    :param name: column name
    :return: cleaned version of the column name
    """
    essential_chars = set(string.ascii_letters + string.digits + "_")
    name_no_spaces = "_".join(name.split())

    return "".join(filter(lambda char: char in set(essential_chars), name_no_spaces))


def subsample_assessments(data_df: pd.DataFrame):
    chosen = []
    for i, assessments in data_df.groupby("installation_id", sort=False):
        chosen.append(dict(assessments.sample(1).iloc[0]))
    return pd.DataFrame(chosen)


def regression_scores_to_pred(
    accuracy_group_proportions: List[float], scores: List[float]
) -> List[int]:
    """
    Matches a list of regression scores to an accuracy group based on desired accuracy group proportions
    :param accuracy_group_proportions: desired accuracy group proportions
    :param scores: regression scores
    :return: a list of predicted accuracy groups
    """
    assert sum(accuracy_group_proportions) == pytest.approx(1.0)

    accumulated_percentile = 0
    bounds = []
    for i in range(3):
        accumulated_percentile += accuracy_group_proportions[i]
        bounds.append(np.percentile(scores, accumulated_percentile * 100))

    def classify(score):
        if score <= bounds[0]:
            return 0
        elif score <= bounds[1]:
            return 1
        elif score <= bounds[2]:
            return 2
        else:
            return 3

    return list(map(classify, scores))
