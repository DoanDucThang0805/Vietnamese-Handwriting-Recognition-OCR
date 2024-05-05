import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from src.Config.config import vocab, Image_Width, Image_Hight, Image_path, Model_path
from src.Loss.CTC_loss import ctc_loss
from src.Model.crnn_model import CRNN_model
from src.Util.util import data, DataGenerator


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
    test_size = 0.2
    numbers_train = int(len(data) * test_size)
    train_set = data.iloc[:numbers_train]
    val_set = data.iloc[numbers_train:]
    x_train = list(train_set.filename)
    y_train = list(train_set.label)
    x_val = list(val_set.filename)
    y_val = list(val_set.label)
    train = DataGenerator(x_train, y_train, 8)
    val = DataGenerator(x_val, y_val, 8)

    model = CRNN_model(input_dim=(Image_Hight, Image_Width, 1), output_dim=len(vocab))
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
