
import logging
import time
import csv
import os
import crypten
import torch
import numpy as np
from examples.meters import AverageMeter


DATASET_DIR = os.path.join(os.path.dirname(__file__), '/dataset')

class DataSetParam:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


class DataSet:
    sonar = DataSetParam('sonar.csv')
    binary_datastet = DataSetParam("binary_datastet.csv")




def load_dataset(dataset):
    dataset_path = os.path.join(DATASET_DIR, dataset)

    # 尝试打开并读取CSV文件
    try:
        with open(dataset_path, 'r', encoding='utf-8') as csvfile:
            spamreader = csv.reader(csvfile)
            data = np.array(list(spamreader))
    except FileNotFoundError:
        print(f"Error: The file {dataset_path} was not found.")
        return None, None
    except Exception as e:
        print(f"Error: An error occurred while reading the file: {e}")
        return None, None

    # 检查数据是否为空
    if data.size == 0:
        print("Error: The dataset is empty.")
        return None, None

    feature = data[:, :-1].astype(np.float64)
    labels = data[:, -1]

    # 创建标签映射
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels], dtype=np.uint8)
    labell = int_labels.reshape(-1,1)
    
    # 将标签转换为独热矩阵
    one_hot_labels = np.zeros((int_labels.size, int_labels.max() + 1), dtype=np.uint8)
    one_hot_labels[np.arange(int_labels.size), int_labels] = 1

    print("======feature_size=====", feature.shape)
    print("======label_size=====", labell.shape)
    return feature, one_hot_labels