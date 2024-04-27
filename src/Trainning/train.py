import numpy as np
from src.Config.config import Max_Length, vocab, Image_With, Image_Hight
from src.Util.util import load_image, encode_to_labels
from src.Util.util import data
from src.Loss.CTC_loss import ctc_loss
from src.Model.crnn_model import CRNN_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


def plot(history):
    # Lấy thông tin từ history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    # Vẽ biểu đồ loss
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("E:/OCR/Image/his_train.png")
    plt.show()


def train():
    # ____________________preprocessing___________________________
    X = load_image()
    labels = list(data.label)
    Y = []
    for label in labels:
        y = encode_to_labels(label)
        padding_length = Max_Length - len(y)
        padding_array = np.zeros(padding_length, dtype=int)
        y = np.concatenate((y, padding_array))
        Y.append(y)
    # ____________________________________________________________

    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(160000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(8)  # helps bottlenecks

    train = dataset.take(int(len(dataset) * .8))
    val = dataset.skip(int(len(dataset) * .8)).take(int(len(dataset) * .1))
    test = dataset.skip(int(len(dataset) * .9)).take(int(len(dataset) * .1))

    model = CRNN_model(input_dim=(Image_Hight, Image_With, 1), output_dim=len(vocab))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=ctc_loss
    )
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=10,
                                   restore_best_weights=True)

    checkpoint = ModelCheckpoint("E:/OCR/Result/model_checkpoint.h5",
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True)

    history = model.fit(
        train,
        epochs=1,
        validation_data=val,
        callbacks=[early_stopping, checkpoint]
    )
    plot(history)
