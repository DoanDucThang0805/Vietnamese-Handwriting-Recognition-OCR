import tensorflow as tf


def ctc_loss(y_true, y_pred):
    """
    Tính toán CTC (Connectionist Temporal Classification) loss.

    Arguments:
    y_true (tensor): Nhãn thật sự, có thể là một SparseTensor.
    y_pred (tensor): Dự đoán của mô hình.

    Returns:
    tensor: Giá trị của hàm mất mát CTC.
    """
    # Tính kích thước batch và độ dài của dự đoán và nhãn
    batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
    input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
    label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

    # Chuyển các kích thước đến dạng tensor
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype='int64')
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype='int64')

    # Tính toán hàm mất mát CTC
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss
