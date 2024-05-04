import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from src.Config.config import Max_Length, vocab, Image_With, Image_Hight, Image_path, Model_path
from src.Util.util import load_image, encode_to_labels, data, DataGenerator
from src.Loss.CTC_loss import ctc_loss
from src.Model.crnn_model import CRNN_model


def plot(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Image_path)
    plt.show()


def train():
    X = load_image()
    labels = list(data.label)
    Y = [np.concatenate((encode_to_labels(label),
                         np.zeros(Max_Length - len(encode_to_labels(label)), dtype=int))) for label in labels]

    # Khởi tạo DataGenerator
    batch_size = 8
    data_generator = DataGenerator(X, Y, batch_size)

    # Tạo Dataset từ generator
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, Image_Hight, Image_With, 1], [None, Max_Length])
    )

    # Các bước xử lý dữ liệu tiếp theo
    dataset = dataset.cache()
    dataset = dataset.shuffle(160000)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Phân chia dữ liệu thành tập train, validation và test
    train_size = int(len(data_generator) * 0.8)
    val_size = int(len(data_generator) * 0.1)
    test_size = len(data_generator) - train_size - val_size

    train = dataset.take(train_size)
    val = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size + val_size).take(test_size)

    model = CRNN_model(input_dim=(Image_Hight, Image_With, 1), output_dim=len(vocab))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=ctc_loss
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(Model_path, monitor='val_loss', save_best_only=True)

    history = model.fit(
        train,
        epochs=1,
        validation_data=val,
        callbacks=[early_stopping, checkpoint]
    )
    plot(history)
