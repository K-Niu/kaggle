import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from dsb_2019.preprocessing.utils import regression_scores_to_pred
from dsb_2019.preprocessing.constants import (
    EVENT_TYPES,
    TITLES,
    WORLDS,
    ASSESSMENTS,
)
from utils.tf import df_to_dataset, dense_to_sparse


MAX_LENGTH = 100
MASK_VALUE = -1.0
SEQUENCE_CATEGORICAL_FEATURES = {
    "types": EVENT_TYPES,
    "titles": TITLES,
    "worlds": WORLDS,
    "days_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}
SEQUENCE_NUMERIC_FEATURES = [
    "hours",
    "times_since_first_game_session",
    "2000_counts",
    "3010_counts",
    "3110_counts",
    "4070_counts",
    "4090_counts",
    "4030_counts",
    "4035_counts",
    "4021_counts",
    "4020_counts",
    "4010_counts",
    "2080_counts",
    "2083_counts",
    "2040_counts",
    "2020_counts",
    "2030_counts",
    "3021_counts",
    "3121_counts",
    "2050_counts",
    "3020_counts",
    "3120_counts",
    "2060_counts",
    "2070_counts",
    "4031_counts",
    "4025_counts",
    "5000_counts",
    "5010_counts",
    "2081_counts",
    "2025_counts",
    "4022_counts",
    "2035_counts",
    "4040_counts",
    "4100_counts",
    "2010_counts",
    "4110_counts",
    "4045_counts",
    "4095_counts",
    "4220_counts",
    "2075_counts",
    "4230_counts",
    "4235_counts",
    "4080_counts",
    "4050_counts",
    "4020_corrects",
    "4100_corrects",
    "4025_corrects",
    "4110_corrects",
    "4020_attempts",
    "4100_attempts",
    "4025_attempts",
    "4110_attempts",
]


def model_fn():
    # Inputs and feature columns
    inputs = {}

    sequence_inputs = {}
    sequence_feature_columns = []
    categorical_embedding_dimensions = {
        "types": 4,
        "titles": 8,
        "worlds": 4,
        "days_of_week": 4,
    }
    for feature in SEQUENCE_CATEGORICAL_FEATURES.keys():
        inputs[feature] = sequence_inputs[feature] = tf.keras.Input(
            shape=(MAX_LENGTH), name=feature, dtype=tf.string
        )
        sequence_feature_columns.append(
            tf.feature_column.embedding_column(
                tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                    feature, SEQUENCE_CATEGORICAL_FEATURES[feature]
                ),
                dimension=categorical_embedding_dimensions[feature],
            )
        )

    for feature in SEQUENCE_NUMERIC_FEATURES:
        inputs[feature] = tf.keras.Input(
            shape=(MAX_LENGTH), name=feature, dtype=tf.float32
        )
        # Numeric inputs to SequenceFeatures must be sparse tensors
        # https://github.com/tensorflow/tensorflow/issues/29879
        sequence_inputs[feature] = tf.keras.layers.Lambda(
            lambda x: dense_to_sparse(x, mask_value=MASK_VALUE)
        )(inputs[feature])
        sequence_feature_columns.append(
            tf.feature_column.sequence_numeric_column(feature)
        )

    sequence_features = tf.keras.experimental.SequenceFeatures(sequence_feature_columns)

    inputs["assessment"] = tf.keras.Input(shape=(), name="assessment", dtype=tf.string)
    assessment_feature_column = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            "assessment", ASSESSMENTS
        )
    )

    assessment_feature = tf.keras.layers.DenseFeatures([assessment_feature_column])

    # Model
    processed_sequence_features, sequence_length = sequence_features(sequence_inputs)
    sequence_length_mask = tf.sequence_mask(sequence_length, maxlen=MAX_LENGTH)
    sequence_lstm = tf.keras.layers.LSTM(100)(
        processed_sequence_features, mask=sequence_length_mask
    )
    sequence_lstm = tf.keras.layers.Dense(32, activation="relu")(sequence_lstm)

    processed_assessments = assessment_feature({"assessment": inputs["assessment"]})

    features_concat = tf.keras.layers.concatenate(
        [sequence_lstm, processed_assessments], axis=1
    )
    dense = tf.keras.layers.Dense(16, activation="relu")(features_concat)
    output = tf.keras.layers.Dense(1, activation="linear")(dense)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
    )

    return model


def train_and_evaluate(features, train_index, val_index, accuracy_group_proportions):
    train, val = features.iloc[train_index], features.iloc[val_index]
    train_dataset = df_to_dataset(
        train.drop(["installation_id"], axis=1),
        batch_size=32,
        target_column="accuracy_group",
        sample_weight_column="sample_weight",
        shuffle=True,
    )
    val_dataset = df_to_dataset(
        val.drop(["installation_id"], axis=1),
        batch_size=len(val),
        target_column="accuracy_group",
        sample_weight_column="sample_weight",
        shuffle=False,
    )
    y_val = val["accuracy_group"]
    val_sample_weights = val["sample_weight"]

    model = model_fn()
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model.fit(
        train_dataset,
        epochs=200,
        validation_data=val_dataset,
        callbacks=[early_stopping_cb],
    )

    val_scores = model.predict(val_dataset)
    val_predicted = regression_scores_to_pred(accuracy_group_proportions, val_scores)

    qwk_score = cohen_kappa_score(
        val_predicted, y_val, weights="quadratic", sample_weight=val_sample_weights
    )
    print(f"Fold QWK score: {qwk_score}")
    print(confusion_matrix(y_val, val_predicted))

    return model, qwk_score


if __name__ == "__main__":
    train_features = pd.read_pickle("dsb_2019/data/sequential_train_features.pkl")
    test_features = pd.read_pickle("dsb_2019/data/sequential_test_features.pkl")

    # Weight train samples by the inverse of how frequently the installation_id appears
    train_features["sample_weight"] = 1 / train_features.groupby("installation_id")[
        "accuracy_group"
    ].transform("count")

    # Estimate accuracy_group proportions in the test set by using sample weights in train instead of sampling
    accuracy_groups = train_features.groupby("accuracy_group")["sample_weight"].agg(
        "sum"
    )
    accuracy_group_proportions = list(accuracy_groups / accuracy_groups.sum())

    # Normalize sequence features
    for feature in SEQUENCE_NUMERIC_FEATURES:
        all_values = []
        for value in train_features[feature]:
            all_values.extend(value)
        value_mean = np.mean(all_values)
        value_std = np.std(all_values)
        train_features[feature] = train_features[feature].apply(
            lambda x: (x - value_mean) / value_std
        )

    # Pad sequences
    for feature in SEQUENCE_CATEGORICAL_FEATURES:
        train_features[feature] = list(
            tf.keras.preprocessing.sequence.pad_sequences(
                train_features[feature],
                padding="post",
                truncating="post",
                dtype=object,
                maxlen=MAX_LENGTH,
                value="",
            )
        )
    for feature in SEQUENCE_NUMERIC_FEATURES:
        train_features[feature] = list(
            tf.keras.preprocessing.sequence.pad_sequences(
                train_features[feature],
                padding="post",
                truncating="post",
                dtype="float32",
                maxlen=MAX_LENGTH,
                value=MASK_VALUE,
            )
        )

    group_kfold = GroupKFold(n_splits=5)
    models = []
    qwk_scores = []
    for train_index, val_index in group_kfold.split(
        train_features, groups=train_features["installation_id"]
    ):
        model, qwk_score = train_and_evaluate(
            train_features, train_index, val_index, accuracy_group_proportions
        )

    print(f"QWK score: {np.mean(qwk_scores)}")

    # Predict on test set
    X_test, installation_ids = (
        test_features.drop(["installation_id"], axis=1),
        test_features["installation_id"],
    )

    model_scores = [model.predict(X_test) for model in models]
    mean_model_scores = np.mean(np.array(model_scores), axis=0)
    final_predictions = pd.Series(
        regression_scores_to_pred(accuracy_group_proportions, list(mean_model_scores))
    )

    submission = pd.DataFrame(
        {"installation_id": installation_ids, "accuracy_group": final_predictions}
    )
    submission.to_csv("submission.csv", index=False)
