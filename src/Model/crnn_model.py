import tensorflow as tf
from keras import Model
from keras.src.layers import Lambda
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, Dense, LSTM


def CRNN_model(input_dim, output_dim, activation="gelu"):
    """
    Xây dựng mô hình CRNN (Convolutional Recurrent Neural Network).

    Arguments:
    input_dim (tuple): Kích thước đầu vào của ảnh (chiều cao, chiều rộng, số kênh).
    output_dim (int): Số lượng lớp đầu ra (bao gồm cả lớp không).
    activation (str): Hàm kích hoạt được sử dụng trong các lớp Conv2D. Mặc định là "gelu".

    Returns:
    model: Mô hình CRNN.
    """
    inputs = Input(shape=input_dim, name="input")

    # Các lớp Convolutional
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation=activation, padding='valid')(inputs)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation=activation, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation=activation, padding='same')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation=activation, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation=activation, padding='same')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=1, activation=activation, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, activation=activation, padding='same')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), strides=1, activation=activation, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 1))(x)

    # Sử dụng Lambda để bọc phép toán tf.squeeze
    squeezed = Lambda(lambda x: tf.squeeze(x, axis=1), name='squeeze')(x)

    # Các lớp Recurrent
    blstm1 = Bidirectional(LSTM(512, return_sequences=True))(squeezed)
    blstm2 = Bidirectional(LSTM(512, return_sequences=True))(blstm1)

    # Lớp Dense đầu ra
    output = Dense(output_dim + 1, activation="softmax", name="output")(blstm2)

    model = Model(inputs=inputs, outputs=output)
    return model

# if __name__ == "__main__":
#     model = CRNN_model((32, 128, 1), 17)
#     print(model.summary())
