import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from src.Config.config import Max_Length, vocab, Image_With, Image_Hight, Image_path, Model_path
from src.Util.util import load_image, encode_to_labels
from src.Util.util import data
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
    Y = []
    for label in labels:
        y = encode_to_labels(label)
        padding_length = Max_Length - len(y)
        padding_array = np.zeros(padding_length, dtype=int)
        y = np.concatenate((y, padding_array))
        Y.append(y)

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: (tf.convert_to_tensor(x), tf.convert_to_tensor(y)))
    dataset = dataset.cache()
    dataset = dataset.shuffle(160000)
    dataset = dataset.batch(32)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/GPU:0"))
    dataset = dataset.prefetch(8)

    train = dataset.take(int(len(dataset) * 0.8))
    val = dataset.skip(int(len(dataset) * 0.8)).take(int(len(dataset) * 0.2))
    test = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))

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
