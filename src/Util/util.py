from src.Prepare_data.preparedata import Preparedata
from src.Config.config import train_path, label_path, test_path, Image_Height, Image_Width, Max_Length, Image_path
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

pre = Preparedata(train_path=train_path, test_path=test_path, label_path=label_path)
data = pre.load_train_data()

labels = list(data.label)
char_list = set()
for label in labels:
    char_list.update(set(label))
vocab = sorted(char_list)
vocab = "".join(vocab)


def load_image(image_path):
    image = tf.io.read_file(image_path)
    decode_image = tf.image.decode_jpeg(contents=image, channels=1)
    convert_image = tf.image.convert_image_dtype(image=decode_image, dtype=tf.float32)
    resize_image = tf.image.resize(images=convert_image, size=(Image_Height, Image_Width))
    image = tf.cast(resize_image, dtype=tf.float32)
    return image


char_to_number = StringLookup(vocabulary=list(vocab), mask_token=None)
number_to_char = StringLookup(vocabulary=char_to_number.get_vocabulary(), mask_token=None, invert=True)


def encode(image_path, label):
    image = load_image(image_path)
    chars = tf.strings.unicode_split(label, input_encoding='UTF-8')
    vector = char_to_number(chars)
    pad_size = Max_Length - tf.shape(vector)[0]
    vector = tf.pad(vector, paddings=[[0, pad_size]], constant_values=len(vocab) + 1)
    return {"image": image, "label": vector}


def decode_pred(pred_label):
    # Input length
    input_len = np.ones(shape=pred_label.shape[0]) * pred_label.shape[1]

    # CTC decode
    decode = keras.backend.ctc_decode(pred_label, input_length=input_len, greedy=True)[0][0][:, :Max_Length]

    # Converting numerics back to their character values
    chars = number_to_char(decode)

    # Join all the characters
    texts = [tf.strings.reduce_join(inputs=char).numpy().decode('UTF-8') for char in chars]

    # Remove the unknown token
    filtered_texts = [text.replace('[UNK]', " ").strip() for text in texts]

    return filtered_texts


def show_images(data, GRID=None, FIGSIZE=(25, 6), cmap='binary_r', model=None, decode_pred=None):
    # Plotting configurations
    if GRID is None:
        GRID = [3, 3]
    plt.figure(figsize=FIGSIZE)
    n_rows, n_cols = GRID

    # Loading Data
    data = next(iter(data))
    images, labels = data['image'], data['label']

    # Ensure you have exactly n_rows * n_cols images to display
    num_images = n_rows * n_cols
    if len(images) < num_images:
        raise ValueError(
            f"Data contains only {len(images)} images, but {num_images} are required for the specified GRID.")

    # Iterate over the data
    for index, (image, label) in enumerate(zip(images[:num_images], labels[:num_images])):

        # Label processing
        text_label = number_to_char(label)
        text_label = tf.strings.reduce_join(text_label).numpy().decode('UTF-8')
        text_label = text_label.replace("[UNK]", " ").strip()

        # Create a sub plot
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(tf.transpose(image, perm=[0, 1, 2]), cmap=cmap)
        plt.axis('off')

        if model is not None and decode_pred is not None:
            # Make prediction
            pred = model.predict(tf.expand_dims(image, axis=0))
            pred = decode_pred(pred)[0]
            title = f"True : {text_label}\nPred : {pred}"
            plt.title(title)
        else:
            # add title
            plt.title(text_label)
    plt.savefig(Image_path, "predict_plot.png")
    plt.show()
