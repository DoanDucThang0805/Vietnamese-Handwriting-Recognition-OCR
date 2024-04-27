import os
import pandas as pd
from src.Config.config import train_path, test_path, label_path, data_path


class Preparedata:
    def __init__(self, train_path, test_path, label_path):
        """
        Khởi tạo một đối tượng Preparedata với các đường dẫn đến thư mục chứa dữ liệu huấn luyện, dữ liệu kiểm tra,
        và nhãn của dữ liệu huấn luyện.

        Parameters:
            train_path (str): Đường dẫn đến thư mục chứa dữ liệu huấn luyện.
            test_path (str): Đường dẫn đến thư mục chứa dữ liệu kiểm tra.
            label_path (str): Đường dẫn đến tệp chứa nhãn của dữ liệu huấn luyện.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.label_path = label_path

    @staticmethod
    def load_train_data():
        """
        Load dữ liệu huấn luyện từ các tệp nhãn và tạo một DataFrame chứa các thông tin về đường dẫn file và nhãn
        tương ứng.

        Returns:
            DataFrame: DataFrame chứa thông tin về đường dẫn file và nhãn của dữ liệu huấn luyện.
        """
        train_data = os.path.join(label_path)
        f = open(train_data, encoding="utf8")
        lines = f.readlines()
        train_labels = dict()
        for line in lines:
            img, label = line.split()
            train_labels[img] = label
        file_path = list(train_labels.keys())
        labels = list(train_labels.values())
        data = list(zip(file_path, labels))
        Dataframe = pd.DataFrame(data, columns=['filename', 'label'])

        # Thêm đường dẫn thư mục chứa dữ liệu huấn luyện vào tên file trong DataFrame
        Dataframe["filename"] = [train_path + f"/{filename}" for filename in Dataframe["filename"]]

        # Lưu DataFrame vào tệp CSV
        csv_file_path = os.path.join(data_path, 'train_data.csv')
        Dataframe.to_csv(csv_file_path, index=False)

        return Dataframe


if __name__ == "__main__":
    pre = Preparedata(train_path, test_path, label_path)
    pre.load_train_data()
