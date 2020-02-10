from typing import List, Tuple

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from dsb_2019.preprocessing.utils import regression_scores_to_pred


def train_and_evaluate(
    features: pd.DataFrame,
    train_index: List[int],
    val_index: List[int],
    accuracy_group_proportions: List[float],
) -> Tuple[lgb.Booster, float]:
    params = {
        "boosting": "gbdt",
        "n_estimators": 5000,
        "objective": "regression",
        "metric": "rmse",
        "lambda_l1": 1,
        "lambda_l2": 1,
        "subsample": 0.75,
        "subsample_freq": 1,
        "feature_fraction": 0.9,
        "learning_rate": 0.01,
        "num_leaves": 50,
        "max_depth": 15,
    }
    train, val = features.iloc[train_index], features.iloc[val_index]
    X_train, y_train, train_sample_weights = (
        train.drop(["accuracy_group", "installation_id", "sample_weight",], axis=1,),
        train["accuracy_group"],
        train["sample_weight"],
    )
    X_val, y_val, val_sample_weights = (
        val.drop(["accuracy_group", "installation_id", "sample_weight",], axis=1,),
        val["accuracy_group"],
        val["sample_weight"],
    )
    train_dataset = lgb.Dataset(
        X_train,
        y_train,
        weight=train_sample_weights,
        categorical_feature=["assessment"],
    )
    val_dataset = lgb.Dataset(
        X_val, y_val, weight=val_sample_weights, categorical_feature=["assessment"]
    )
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=10000,
        early_stopping_rounds=100,
        valid_sets=[train_dataset, val_dataset],
        verbose_eval=100,
    )

    val_scores = model.predict(X_val)
    val_predicted = regression_scores_to_pred(accuracy_group_proportions, val_scores)

    qwk_score = cohen_kappa_score(
        val_predicted, y_val, weights="quadratic", sample_weight=val_sample_weights
    )
    print(f"Fold QWK score: {qwk_score}")
    print(confusion_matrix(y_val, val_predicted))

    return model, qwk_score


if __name__ == "__main__":
    train_features = pd.read_csv("dsb_2019/data/flat_train_features.csv")
    test_features = pd.read_csv("dsb_2019/data/flat_test_features.csv")

    # Weight train samples by the inverse of how frequently the installation_id appears
    train_features["sample_weight"] = 1 / train_features.groupby("installation_id")[
        "accuracy_group"
    ].transform("count")
    # Estimate accuracy_group proportions in the test set by using sample weights in train instead of sampling
    accuracy_groups = train_features.groupby("accuracy_group")["sample_weight"].agg(
        "sum"
    )
    accuracy_group_proportions = list(accuracy_groups / accuracy_groups.sum())

    assessment_encoder = LabelEncoder()
    train_features["assessment"] = assessment_encoder.fit_transform(
        train_features["assessment"]
    )

    group_kfold = GroupKFold(n_splits=5)
    models = []
    qwk_scores = []
    for train_index, val_index in group_kfold.split(
        train_features, groups=train_features["installation_id"]
    ):
        model, qwk_score = train_and_evaluate(
            train_features,
            list(train_index),
            list(val_index),
            accuracy_group_proportions,
        )
        models.append(model)
        qwk_scores.append(qwk_score)

    print(f"QWK score: {np.mean(qwk_scores)}")

    # Predict on test set
    X_test, installation_ids = (
        test_features.drop(["installation_id"], axis=1),
        test_features["installation_id"],
    )
    X_test["assessment"] = assessment_encoder.transform(X_test["assessment"])

    model_scores = [model.predict(X_test) for model in models]
    mean_model_scores = np.mean(np.array(model_scores), axis=0)
    final_predictions = pd.Series(
        regression_scores_to_pred(accuracy_group_proportions, list(mean_model_scores))
    )

    submission = pd.DataFrame(
        {"installation_id": installation_ids, "accuracy_group": final_predictions}
    )
    submission.to_csv("submission.csv", index=False)
