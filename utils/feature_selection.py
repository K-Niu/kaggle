from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def remove_correlated_features(
    features_df: pd.DataFrame, feature_names: List[str], cutoff: float = 0.99
) -> List[str]:
    """
    Finds a list of correlated features to remove by doing pairwise correlation calculations
    :param features_df: a features DataFrame
    :param feature_names: feature column names
    :param cutoff: maximum correlation coefficient
    :return: column names to remove
    """
    to_remove = set([])
    for i, feat_a in enumerate(feature_names):
        for j, feat_b in enumerate(feature_names):
            if i >= j or feat_a in to_remove or feat_b in to_remove:
                continue
            correlation = np.corrcoef(features_df[feat_a], features_df[feat_b])[0][1]
            if correlation > cutoff:
                to_remove.add(feat_b)

    return list(to_remove)


def remove_high_entropy_features(
    features_df: pd.DataFrame, feature_names: List[str], cutoff: float = 0.9
) -> List[str]:
    """
    Finds a list of high entropy features to remove
    :param features_df: a features DataFrame
    :param feature_names: feature column names
    :param cutoff: maximum proportion of the maximum possible entropy
    :return: column names to remove
    """
    to_remove = []
    max_entropy = stats.entropy(np.ones(len(features_df)))
    for feat in feature_names:
        feat_entropy = stats.entropy(features_df[feat])
        if np.isnan(feat_entropy) or feat_entropy > (cutoff * max_entropy):
            to_remove.append(feat)

    return to_remove
