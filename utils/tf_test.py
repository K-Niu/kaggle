import tensorflow as tf
import pandas as pd

from utils.tf import dense_to_sparse, df_to_dataset


def test_dense_to_sparse():
    dense_tensor = tf.constant([[1.0, 2.0, 3.0], [-1.0, 5.0, -1.0]])
    assert tf.reduce_all(
        tf.equal(
            tf.sparse.to_dense(
                dense_to_sparse(dense_tensor, mask_value=-1.0), default_value=-1.0
            ),
            dense_tensor,
        )
    )


def test_df_to_dataset():
    regression_data = pd.DataFrame(
        [
            {"feat_1": 1.0, "feat_2": 2.0, "label": 2.3, "sample_weight": 1.0},
            {"feat_1": 2.0, "feat_2": 3.0, "label": 3.4, "sample_weight": 0.5},
            {"feat_1": 3.0, "feat_2": 4.0, "label": 4.5, "sample_weight": 0.3},
        ]
    )
    classification_data = pd.DataFrame(
        [
            {"feat_1": 1.0, "feat_2": 2.0, "label": 0, "sample_weight": 1.0},
            {"feat_1": 2.0, "feat_2": 3.0, "label": 1, "sample_weight": 0.5},
            {"feat_1": 3.0, "feat_2": 4.0, "label": 2, "sample_weight": 0.3},
        ]
    )

    regression_ds = df_to_dataset(
        regression_data, batch_size=3, target_column="label", shuffle=False
    )
    regression_weighted_ds = df_to_dataset(
        regression_data,
        batch_size=3,
        target_column="label",
        sample_weight_column="sample_weight",
        shuffle=False,
    )
    classification_ds = df_to_dataset(
        classification_data,
        batch_size=3,
        target_column="label",
        num_classes=3,
        shuffle=False,
    )
    classification_weighted_ds = df_to_dataset(
        classification_data,
        batch_size=3,
        target_column="label",
        sample_weight_column="sample_weight",
        num_classes=3,
        shuffle=False,
    )

    for batch in regression_ds.take(1):
        features, labels = batch
        assert tf.reduce_all(
            tf.equal(features["feat_1"], tf.constant([1.0, 2.0, 3.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(features["feat_2"], tf.constant([2.0, 3.0, 4.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(labels, tf.constant([2.3, 3.4, 4.5], dtype=tf.float64))
        )
        break

    for batch in regression_weighted_ds.take(1):
        features, labels, sample_weights = batch
        assert tf.reduce_all(
            tf.equal(features["feat_1"], tf.constant([1.0, 2.0, 3.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(features["feat_2"], tf.constant([2.0, 3.0, 4.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(labels, tf.constant([2.3, 3.4, 4.5], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(sample_weights, tf.constant([1.0, 0.5, 0.3], dtype=tf.float64))
        )
        break

    for batch in classification_ds.take(1):
        features, labels = batch
        assert tf.reduce_all(
            tf.equal(features["feat_1"], tf.constant([1.0, 2.0, 3.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(features["feat_2"], tf.constant([2.0, 3.0, 4.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(
                labels, tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
            )
        )
        break

    for batch in classification_weighted_ds.take(1):
        features, labels, sample_weights = batch
        assert tf.reduce_all(
            tf.equal(features["feat_1"], tf.constant([1.0, 2.0, 3.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(features["feat_2"], tf.constant([2.0, 3.0, 4.0], dtype=tf.float64))
        )
        assert tf.reduce_all(
            tf.equal(
                labels, tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
            )
        )
        assert tf.reduce_all(
            tf.equal(sample_weights, tf.constant([1.0, 0.5, 0.3], dtype=tf.float64))
        )
        break
