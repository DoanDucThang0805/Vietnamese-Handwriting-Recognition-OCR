import tensorflow as tf
from keras import Model
from keras.src.layers import Lambda
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Bidirectional, Dense, LSTM
from src.Config.config import Image_Height, Image_Width
from src.Loss.CTC_loss import CTCLayer
from src.Util.util import char_to_number


def CRNN_model(activation="relu", output_dim=len(char_to_number.get_vocabulary()) + 1):
    images = Input(shape=(Image_Height, Image_Width, 1), name="image")
    labels = Input(shape=(None,), name="label")

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=1, activation=activation, padding='same',
               kernel_initializer=he_normal())(images)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation=activation, padding='same',
               kernel_initializer=he_normal())(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation=activation, padding='same',
               kernel_initializer=he_normal())(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 1))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=1, activation=activation, padding='same',
               kernel_initializer=he_normal())(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 1))(x)

    x = Lambda(lambda x: tf.squeeze(x, axis=1), name='squeeze')(x)

    blstm_1 = Bidirectional(LSTM(128, return_sequences=True))(x)
    blstm_2 = Bidirectional(LSTM(64, return_sequences=True))(blstm_1)

    output = Dense(output_dim, activation="softmax", name="output")(blstm_2)

    ctc_layer = CTCLayer()(labels, output)

    model = Model(inputs=[images, labels], outputs=[ctc_layer], name="CRNN_Model")
    return model


if __name__ == "__main__":
    model = CRNN_model()
    print(model.summary())
