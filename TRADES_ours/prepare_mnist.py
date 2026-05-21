"""
手动处理 MNIST 原始数据，生成 torchvision 可用的 .pt 文件。

背景：TRADES 项目依赖的 torchvision 0.2.1 中的 MNIST 下载链接已失效（404），
因此需要手动下载原始 .gz 文件并解压，再用此脚本生成 processed/training.pt 和 processed/test.pt。
"""

import os
import codecs
import torch
import numpy as np


def get_int(b):
    """将4字节二进制数据解析为整数（大端序）。"""
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    """读取 MNIST 标签文件。

    文件格式：
    - 偏移 0-3:   magic number (2049)
    - 偏移 4-7:   标签数量
    - 偏移 8+:    每个标签 1 字节 (uint8)
    """
    with open(path, "rb") as f:
        data = f.read()
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    """读取 MNIST 图片文件。

    文件格式：
    - 偏移 0-3:   magic number (2051)
    - 偏移 4-7:   图片数量
    - 偏移 8-11:  图片高度 (28)
    - 偏移 12-15: 图片宽度 (28)
    - 偏移 16+:   每个像素 1 字节 (uint8)，按行优先展开
    """
    with open(path, "rb") as f:
        data = f.read()
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def main():
    # 数据根目录（相对于 TRADES 项目目录为 ../data）
    # torchvision 的 MNIST 类直接在此目录下创建 raw/ 和 processed/ 子目录
    root = os.path.join(os.path.dirname(__file__), "..", "data")
    root = os.path.abspath(root)

    raw_dir = os.path.join(root, "raw")
    proc_dir = os.path.join(root, "processed")
    os.makedirs(proc_dir, exist_ok=True)

    print(f"Root dir: {root}")
    print(f"Raw dir:  {raw_dir}")
    print(f"Proc dir: {proc_dir}")

    # 读取原始二进制文件
    print("Reading train-images-idx3-ubyte ...")
    train_images = read_image_file(os.path.join(raw_dir, "train-images-idx3-ubyte"))
    print(f"  shape: {train_images.shape}")

    print("Reading train-labels-idx1-ubyte ...")
    train_labels = read_label_file(os.path.join(raw_dir, "train-labels-idx1-ubyte"))
    print(f"  shape: {train_labels.shape}")

    print("Reading t10k-images-idx3-ubyte ...")
    test_images = read_image_file(os.path.join(raw_dir, "t10k-images-idx3-ubyte"))
    print(f"  shape: {test_images.shape}")

    print("Reading t10k-labels-idx1-ubyte ...")
    test_labels = read_label_file(os.path.join(raw_dir, "t10k-labels-idx1-ubyte"))
    print(f"  shape: {test_labels.shape}")

    # 打包并保存为 torch .pt 格式
    training_set = (train_images, train_labels)
    test_set = (test_images, test_labels)

    training_path = os.path.join(proc_dir, "training.pt")
    test_path = os.path.join(proc_dir, "test.pt")

    print(f"Saving {training_path} ...")
    with open(training_path, "wb") as f:
        torch.save(training_set, f)

    print(f"Saving {test_path} ...")
    with open(test_path, "wb") as f:
        torch.save(test_set, f)

    print("Done! MNIST dataset is ready.")


if __name__ == "__main__":
    main()
