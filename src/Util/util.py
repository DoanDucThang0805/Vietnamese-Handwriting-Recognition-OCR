import cv2
from keras.utils import Sequence
from src.Config.config import Image_Width, Image_Hight, Max_Length
from src.Prepare_data.preparedata import Preparedata
import numpy as np
from itertools import groupby

# Xây dựng từ điển theo các kí tự có trong label
data = Preparedata.load_train_data()
labels = list(data.label)
char_list = set()
for label in labels:
    char_list.update(set(label))
vocab = sorted(char_list)
vocab = "".join(vocab)


def encode_to_labels(text):
    """
    Chuyển đổi văn bản thành danh sách các chỉ mục tương ứng trong từ vựng.

    Parameters:
        text (str): Chuỗi văn bản cần được mã hóa.

    Returns:
        list: Danh sách các chỉ mục tương ứng với các ký tự trong văn bản.
    """
    index_list = []  # Khởi tạo danh sách để lưu trữ chỉ mục của các ký tự trong văn bản
    for index, char in enumerate(text):
        try:
            index_list.append(vocab.index(char))  # Tìm chỉ mục của ký tự trong từ vựng và thêm vào danh sách
        except ValueError:
            print("Không tìm thấy trong từ vựng: ", char)  # In ra thông báo nếu ký tự không có trong từ vựng
    return index_list  # Trả về danh sách các chỉ mục tương ứng với các ký tự trong văn bản


class DataGenerator(Sequence):
    """
    Lớp DataGenerator dùng để tạo các batch dữ liệu từ tập dữ liệu ảnh và nhãn tương ứng.

    Arguments:
        x_set (list): Danh sách đường dẫn đến các tệp ảnh.
        y_set (list): Danh sách các nhãn tương ứng với từng ảnh.
        batch_size (int): Kích thước của mỗi batch dữ liệu.

    Methods:
        __len__: Trả về số lượng batch cần thiết để cover toàn bộ tập dữ liệu.
        __getitem__: Lấy một batch dữ liệu dựa trên chỉ mục.
        data_generation: Tạo dữ liệu cho một batch.
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        Khởi tạo một đối tượng DataGenerator.

        Parameters:
            x_set (list): Danh sách đường dẫn đến các tệp ảnh.
            y_set (list): Danh sách các nhãn tương ứng với từng ảnh.
            batch_size (int): Kích thước của mỗi batch dữ liệu.
        """
        super().__init__()  # Khởi tạo lớp cơ sở Sequence
        self.x = x_set  # Dữ liệu đầu vào (danh sách các đường dẫn tệp ảnh)
        self.y = y_set  # Dữ liệu đầu ra (danh sách các nhãn tương ứng)
        self.batch_size = batch_size  # Kích thước batch cho việc tạo dữ liệu

    def __len__(self):
        """
        Trả về số lượng batch cần thiết để cover toàn bộ tập dữ liệu.

        Returns:
            int: Số lượng batch cần thiết.
        """
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Lấy một batch dữ liệu dựa trên chỉ mục.

        Parameters:
            idx (int): Chỉ mục của batch cần lấy.

        Returns:
            tuple: Một tuple chứa các mảng đầu vào và đầu ra của batch.
        """
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]  # Batch dữ liệu đầu vào
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]  # Batch dữ liệu đầu ra
        X, Y = self.data_generation(batch_x, batch_y)  # Tạo dữ liệu cho batch
        return X, Y

    @staticmethod
    def data_generation(batch_x, batch_y):
        """
        Tạo dữ liệu cho một batch.

        Parameters:
            batch_x (list): Danh sách đường dẫn đến các tệp ảnh trong batch.
            batch_y (list): Danh sách các nhãn tương ứng với từng ảnh trong batch.

        Returns:
            tuple: Một tuple chứa các mảng đầu vào và đầu ra của batch.
        """
        # Khởi tạo mảng để chứa ảnh đầu vào và nhãn đầu ra
        X = np.ones([len(batch_x), Image_Hight, Image_Width, 1])  # Mảng cho ảnh đầu vào
        Y = np.ones([len(batch_x), Max_Length])  # Mảng cho nhãn đầu ra

        # Xử lý từng ảnh trong batch
        for i, img in enumerate(batch_x):
            # Đọc ảnh, chuyển đổi thành ảnh xám, và resize ảnh về kích thước yêu cầu
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (Image_Width, Image_Hight))
            image = np.expand_dims(image, -1)  # Thêm một chiều để phù hợp với hình dạng đầu vào
            X[i] = image  # Gán ảnh đã xử lý vào mảng batch

        # Xử lý từng nhãn văn bản trong batch
        for i, text in enumerate(batch_y):
            # Mã hóa nhãn văn bản thành một chuỗi số nguyên
            text_encoded = encode_to_labels(text)
            # Thêm các số 0 vào chuỗi đã mã hóa để có độ dài tối đa
            Y[i] = text_encoded + [0 for _ in range(Max_Length - len(text_encoded))]

        return X, Y


def ctc_decoder(preds, blank_label=0):
    """
    Giải mã dự đoán từ mô hình CTC thành văn bản.

    Tham số:
    preds: Dự đoán từ mô hình CTC (mảng numpy).
    blank_label: Chỉ mục nhãn đại diện cho ký hiệu trắng. Mặc định là 0.

    Returns:
    decoded_texts: Danh sách các văn bản đã được giải mã.
    """
    decoded_texts = []
    for pred in preds:
        # Loại bỏ các nhãn trắng và nhãn liên tiếp trùng nhau
        decoded = [label for label, _ in groupby(pred) if label != blank_label]
        # Loại bỏ nhãn trắng nếu có
        decoded = [label for label in decoded if label != blank_label]
        # Chuyển đổi chỉ mục nhãn thành ký tự
        decoded_text = ''.join([chr(label + ord('a')) for label in decoded])
        decoded_texts.append(decoded_text)
    return decoded_texts
