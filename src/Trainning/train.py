from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.data as tfd
from src.Util.util import data, encode, decode_pred, show_images
from src.Config.config import Batch_Size, Image_path, Model_path, Image_Height, Image_Width
from src.Model.crnn_model import CRNN_model
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

AUTOTUNE = tfd.AUTOTUNE
train_df, test = train_test_split(data, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test, test_size=0.5, random_state=42)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (np.array(train_df['filename'].to_list()), np.array(train_df['label'].to_list()))).shuffle(1000).map(encode,
                                                                                                         num_parallel_calls=AUTOTUNE).batch(
    Batch_Size).prefetch(AUTOTUNE)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (np.array(val_df['filename'].to_list()), np.array(val_df['label'].to_list()))).shuffle(1000).map(encode,
                                                                                                     num_parallel_calls=AUTOTUNE).batch(
    Batch_Size).prefetch(AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (np.array(test_df['filename'].to_list()), np.array(test_df['label'].to_list()))).shuffle(1000).map(encode,
                                                                                                       num_parallel_calls=AUTOTUNE).batch(
    Batch_Size).prefetch(AUTOTUNE)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(Model_path, monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6, verbose=1)
images = Input(shape=(Image_Height, Image_Width, 1), name="image")


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
    plt.savefig(Image_path, "his_train.png")
    plt.show()


def train():
    model = CRNN_model()
    model.compile(optimizer=Adam(learning_rate=0.001))
    history = model.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )
    plot(history)
    ocr_model = Model(
        inputs=images,
        outputs=model.get_layer(name="output").output
    )
    y_pred = ocr_model.predict(test_dataset)
    y_pred = decode_pred(y_pred)
    show_images(data=y_pred, model=ocr_model, decode_pred=decode_pred(y_pred))
