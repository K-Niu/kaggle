from typing import Optional

import tensorflow as tf
import pandas as pd


@tf.function
def quad_cohen_kappa_loss(
    y_true, y_pred, row_label_vec, col_label_vec, weight_mat, eps=1e-6
):
    batch_size, num_classes = tf.shape(y_true)[0], tf.shape(y_true)[1]
    labels = tf.matmul(y_true, col_label_vec)
    weight = tf.pow(
        tf.tile(labels, [1, num_classes]) - tf.tile(row_label_vec, [batch_size, 1]), 2
    )
    weight /= tf.cast(tf.pow(num_classes - 1, 2), dtype=tf.float64)

    numerator = tf.reduce_sum(weight * y_pred)
    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(
                weight_mat, tf.transpose(tf.reduce_sum(y_pred, axis=0, keepdims=True))
            ),
        )
    )
    denominator /= tf.cast(batch_size, dtype=tf.float64)

    return tf.math.log(numerator / denominator + eps)


class QuadCohenKappaLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, name="cohen_kappa_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)

        self.num_classes = num_classes
        label_vec = tf.range(num_classes, dtype=tf.float64)
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        self.weight_mat = tf.pow(
            tf.tile(self.row_label_vec, [num_classes, 1])
            - tf.tile(self.col_label_vec, [1, num_classes]),
            2,
        )
        self.weight_mat /= tf.cast(tf.pow(num_classes - 1, 2), dtype=tf.float64)

    def call(self, y_true, y_pred, sample_weight=None):
        # row_label_vec, col_label_vec, weight_mat are calculated outside of quad_cohen_kappa_loss
        # to reduce the number of computations because they are batch agnostic
        return quad_cohen_kappa_loss(
            y_true, y_pred, self.row_label_vec, self.col_label_vec, self.weight_mat
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"num_classes": self.num_classes}
        return dict(list(base_config.items()) + list(config.items()))


def dense_to_sparse(dense_tensor: tf.Tensor, mask_value: float) -> tf.SparseTensor:
    """
    Used to prepare dense numeric tensors for tf.keras.experimental.SequenceFeatures.
    Sequence numeric column inputs to SequenceFeatures must be sparse tensors:
    https://github.com/tensorflow/tensorflow/issues/29879
    :param dense_tensor: dense tensor
    :param mask_value: value indicating padding in a sequence
    :return: sparse tensor
    """
    # We include the indices of 0 values because
    # otherwise SequenceFeatures doesn't count them in the sequence length
    # https://github.com/tensorflow/tensorflow/issues/27442
    indices = tf.where(
        tf.not_equal(dense_tensor, tf.constant(mask_value, dtype=tf.float32))
    )
    values = tf.gather_nd(dense_tensor, indices)

    return tf.SparseTensor(
        indices, values, tf.cast(tf.shape(dense_tensor), dtype=tf.int64)
    )


def df_to_dataset(
    df: pd.DataFrame,
    batch_size: int,
    target_column: str,
    sample_weight_column: Optional[str] = None,
    num_classes: Optional[int] = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    df = df.copy()
    labels = df.pop(target_column)
    if num_classes and num_classes > 2:
        labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
    if sample_weight_column:
        sample_weights = df.pop(sample_weight_column)
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels, sample_weights))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size)
    return dataset
