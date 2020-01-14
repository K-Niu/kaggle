import json
import pandas as pd
import numpy as np
import tensorflow as tf


with open("event_types.json", "r") as f:
    EVENT_TYPES = json.loads(f.read())
with open("event_codes.json", "r") as f:
    EVENT_CODES = json.loads(f.read())
with open("titles.json", "r") as f:
    TITLES = json.loads(f.read())
with open("worlds.json", "r") as f:
    WORLDS = json.loads(f.read())
with open("measurement_event_codes.json", "r") as f:
    MEASUREMENT_EVENT_CODES = json.loads(f.read())

MAX_LENGTH = 100
MASK_VALUE = -1.0
SEQUENCE_CATEGORICAL_FEATURES = {
    "types": EVENT_TYPES,
    "titles": TITLES,
    "worlds": WORLDS,
    "days_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
}
# SEQUENCE_NUMERIC_FEATURES = ["hours", "times_since_first_game_session",
#                       "2000_counts", "3010_counts", "3110_counts", "4070_counts",
#                       "4090_counts", "4030_counts", "4035_counts", "4021_counts",
#                       "4020_counts", "4010_counts", "2080_counts", "2083_counts",
#                       "2040_counts", "2020_counts", "2030_counts", "3021_counts",
#                       "3121_counts", "2050_counts", "3020_counts", "3120_counts",
#                       "2060_counts", "2070_counts", "4031_counts", "4025_counts",
#                       "5000_counts", "5010_counts", "2081_counts", "2025_counts",
#                       "4022_counts", "2035_counts", "4040_counts", "4100_counts",
#                       "2010_counts", "4110_counts", "4045_counts", "4095_counts",
#                       "4220_counts", "2075_counts", "4230_counts", "4235_counts",
#                       "4080_counts", "4050_counts", "4020_succeses", "4100_succeses",
#                       "4025_succeses", "4110_succeses", "4020_attempts", "4100_attempts",
#                       "4025_attempts", "4110_attempts"]
SEQUENCE_NUMERIC_FEATURES = ["hours", "times_since_first_game_session", "4020_succeses", "4100_succeses",
                      "4025_succeses", "4110_succeses", "4020_attempts", "4100_attempts",
                      "4025_attempts", "4110_attempts"]

features = pd.read_pickle("train_features.pkl")
features = features[
    list(SEQUENCE_CATEGORICAL_FEATURES.keys()) + SEQUENCE_NUMERIC_FEATURES + ["installation_id", "assessment", "accuracy_group"]
]
features["sample_weight"] = 1 / features.groupby("installation_id")["accuracy_group"].transform("count")

# Estimate accuracy_group proportions in the test set by using sample weights in train instead of sampling
accuracy_groups = features.groupby("accuracy_group")["sample_weight"].agg("sum")
accuracy_group_proportions = accuracy_groups / accuracy_groups.sum()


def scores_to_pred(scores):
    accumulated_percentile = 0
    bounds = []
    for i in range(3):
        accumulated_percentile += accuracy_group_proportions[i]
        bounds.append(np.percentile(scores, accumulated_percentile * 100))

    print(f"Cutoff bounds: {bounds}")

    def classify(score):
        if score <= bounds[0]:
            return 0
        elif score <= bounds[1]:
            return 1
        elif score <= bounds[2]:
            return 2
        else:
            return 3

    return np.array(list(map(classify, scores)))
# Normalize sequence features

# Pad sequences
for feature in SEQUENCE_CATEGORICAL_FEATURES:
    features[feature] = list(tf.keras.preprocessing.sequence.pad_sequences(
        features[feature],
        padding="post",
        truncating="post",
        dtype=object,
        maxlen=MAX_LENGTH,
        value=""
    ))
for feature in SEQUENCE_NUMERIC_FEATURES:
    features[feature] = list(tf.keras.preprocessing.sequence.pad_sequences(
        features[feature],
        padding="post",
        truncating="post",
        dtype="float32",
        maxlen=MAX_LENGTH,
        value=MASK_VALUE
    ))



def dense_to_sparse(dense_tensor):
    # We include the indices of 0 values because
    # otherwise SequenceFeatures doesn't count them in the sequence length
    # https://github.com/tensorflow/tensorflow/issues/27442
    indices = tf.where(tf.not_equal(dense_tensor, tf.constant(MASK_VALUE, dtype=tf.float32)))
    values = tf.gather_nd(dense_tensor, indices)

    return tf.SparseTensor(indices, values, tf.cast(tf.shape(dense_tensor), dtype=tf.int64))


def df_to_dataset(dataframe, batch_size=32, shuffle=True):
    dataframe = dataframe.copy()
    labels = dataframe.pop("accuracy_group")
    sample_weights = dataframe.pop("sample_weight")
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels, sample_weights))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size)
    return dataset


train = df_to_dataset(features.iloc[:32].drop(["installation_id"], axis=1))
for example in train.take(1):
    break


sequence_inputs = {}
sequence_feature_columns = []
categorical_embedding_dimensions = {
    "types": 4,
    "titles": 8,
    "worlds": 4,
    "days_of_week": 4
}
for feature in ["types"]:
    sequence_inputs[feature] = example[0][feature]
    sequence_feature_columns.append(
        tf.feature_column.embedding_column(
            tf.feature_column.sequence_categorical_column_with_vocabulary_list(
                feature,
                SEQUENCE_CATEGORICAL_FEATURES[feature]
            ),
            dimension=categorical_embedding_dimensions[feature]
        )
    )
for feature in ["hours", "4020_succeses"]:
    sequence_inputs[feature] = tf.keras.layers.Lambda(lambda x: dense_to_sparse(x))(example[0][feature])
    sequence_feature_columns.append(tf.feature_column.sequence_numeric_column(feature))

sequence_features = tf.keras.experimental.SequenceFeatures(sequence_feature_columns)
processed_sequence_features, sequence_length = sequence_features(sequence_inputs)
sequence_length_mask = tf.sequence_mask(sequence_length, maxlen=MAX_LENGTH)
sequence_lstm = tf.keras.layers.LSTM(100)(processed_sequence_features, mask=sequence_length_mask)

print("hi")