import os
import shutil
from random import shuffle

def split_data(source_folder, dest_folder, train_ratio=0.9):
    # Đảm bảo rằng thư mục đích tồn tại
    train_folder = os.path.join(dest_folder, 'Train')
    test_folder = os.path.join(dest_folder, 'Test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Duyệt qua mỗi thư mục nhãn trong thư mục nguồn
    labels = []
    with open("group_tu2.txt","r",encoding="utf8") as pr:
        labels = [x.split("\n")[0] for x in pr.readline()]
    for label in os.listdir(source_folder):
        label_dir = os.path.join(source_folder, label)
        if os.path.isdir(label_dir):
            videos = os.listdir(label_dir)
            shuffle(videos)  # Xáo trộn danh sách video

            # Phân chia video thành hai phần: train và test
            split_index = int(len(videos) * train_ratio)
            train_videos = videos[:split_index]
            test_videos = videos[split_index:]

            # Tạo thư mục cho nhãn trong train và test
            train_label_dir = os.path.join(train_folder, label)
            test_label_dir = os.path.join(test_folder, label)

            if not os.path.exists(train_label_dir):
                os.makedirs(train_label_dir)
            if not os.path.exists(test_label_dir):
                os.makedirs(test_label_dir)

            # Sao chép video vào thư mục tương ứng
            for video in train_videos:
                shutil.copy(os.path.join(label_dir, video), train_label_dir)
            for video in test_videos:
                shutil.copy(os.path.join(label_dir, video), test_label_dir)

source_folder = 'data'
dest_folder = 'data_split'
split_data(source_folder, dest_folder)
