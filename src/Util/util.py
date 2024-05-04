import numpy as np
from src.Prepare_data.preparedata import Preparedata
from src.Config.config import Image_With, Image_Hight
import cv2
from keras.utils import Sequence

# Xây dựng từ điển theo các kí tự có trong label
data = Preparedata.load_train_data()
labels = list(data.label)
char_list = set()
for label in labels:
    char_list.update(set(label))
vocab = sorted(char_list)
vocab = "".join(vocab)


def encode_to_labels(text):
    index_list = []
    for index, char in enumerate(text):
        try:
            index_list.append(vocab.index(char))
        except ValueError:
            print("No found in vocab: ", char)
    return index_list


image_path = list(data.filename)


def load_image():
    image_list = []
    for path in image_path:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (Image_With, Image_Hight))
        image = np.expand_dims(image, -1)
        image = image / 255
        image_list.append(image)
    return np.array(image_list)


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        super().__init__()
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
