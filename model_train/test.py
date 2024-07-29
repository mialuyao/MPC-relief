
import logging
import time
import csv
import os
import crypten
import torch
import numpy as np
from examples.meters import AverageMeter


DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset')

class DataSetParam:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


class DataSet:
    sonar = DataSetParam('sonar.csv')
    sonar_selected_55 = DataSetParam('sonar_selected_55.csv')
    sonar_selected_50 = DataSetParam('sonar_selected_50.csv')
    sonar_selected_48 = DataSetParam('sonar_selected_48.csv')
    sonar_selected_45 = DataSetParam('sonar_selected_45.csv')
    sonar_selected_40 = DataSetParam('sonar_selected_40.csv')
    
    leukemia = DataSetParam("leukemia.csv")
    leukemia_selected_half = DataSetParam("leukemia_selected_half.csv")
    leukemia_selected_60 = DataSetParam("leukemia_selected_60%.csv")   
    leukemia_selected_70 = DataSetParam("leukemia_selected_70%.csv")
    leukemia_selected_80 = DataSetParam("leukemia_selected_80%.csv")
    leukemia_selected_90 = DataSetParam("leukemia_selected_90%.csv")
    
    colon = DataSetParam("colon.csv")
    colon_selected_half = DataSetParam("colon_selected_half.csv")
    colon_selected_60 = DataSetParam("colon_selected_60%.csv")   
    colon_selected_70 = DataSetParam("colon_selected_70%.csv")
    colon_selected_80 = DataSetParam("colon_selected_80%.csv")
    colon_selected_90 = DataSetParam("colon_selected_90%.csv")
    




def train_linear_svm(features, labels, epochs, lr, print_time=False):
    # Initialize random weights
    w = features.new(torch.randn(features.size(1),1))
    b = features.new(torch.randn(1))
    # print("==========w========", w.shape)
    # print("==========b========", b.shape)
    # print("=======features=====", features.shape)

    # if print_time:
    #     pt_time = AverageMeter()
    #     end = time.time()
    
    filename = "Accuracy.txt"
    with open(filename, 'a') as f:
        for epoch in range(epochs):
            # Forward
            label_predictions = features.matmul(w).add(b).sign()
            # label_predictions = w.matmul(features.T).add(b).sign()
            # print("=======labels======", labels.shape)
            # print("=======label_predictions======", label_predictions.shape)
            # Compute accuracy
            correct = label_predictions.mul(labels.view(-1,1))
            # print("=======correct======", correct.shape)
            accuracy = correct.add(1).div(2).mean()
            if crypten.is_encrypted_tensor(accuracy):
                accuracy = accuracy.get_plain_text()

            # Print Accuracy once
            if crypten.communicator.get().get_rank() == 0:
                # print(
                #     f"Epoch {epoch} --- Training Accuracy %.2f%%" % (accuracy.item() * 100)
                # )
                accuracy_str = "%.2f%%" % (accuracy * 100)
                accuracy_str_without_percent = accuracy_str.replace('%', '')
                f.write(accuracy_str_without_percent + ",")

            # Backward
            loss_grad = -labels * (1 - correct) * 0.5  # Hinge loss
            # print("=======loss_grad======\n", loss_grad.get_plain_text())
            b_grad = loss_grad.mean()
            # print("=======b_grad ======", b_grad.shape)
            w_grad = features.t().matmul(loss_grad).div(loss_grad.size(0))
            # print("=======w_grad ======", w_grad.shape)
            

            # Update
            w -= w_grad * lr
            b -= b_grad * lr

            # if print_time:
            #     iter_time = time.time() - end
            #     pt_time.add(iter_time)
            #     print("Time %.6f (%.6f)" % (iter_time, pt_time.value()))
            #     end = time.time()

    return w, b


def evaluate_linear_svm(features, labels, w, b):
    """Compute accuracy on a test set"""
    predictions = w.matmul(features).add(b).sign()
    correct = predictions.mul(labels)
    accuracy = correct.add(1).div(2).mean().get_plain_text()
    if crypten.communicator.get().get_rank() == 0:
        # print("Test accuracy %.2f%%" % (accuracy.item() * 100))
        accuracy_str = "%.2f%%" % (accuracy * 100)
        filename = "Accuracy.txt"
        with open(filename, 'w') as f:
            f.write(accuracy_str)


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

    # print("======feature_size=====", feature.shape)
    # print("======label_size=====", labell.shape)
    return feature, labell



def run_mpc_linear_svm(
    epochs=100, lr=0.5
):
    crypten.init()

    # Set random seed for reproducibility
    torch.manual_seed(1)

    data_train = DataSet.colon_selected_90
    
    x, y = load_dataset(data_train.dataset_name)

# NOTE：crypten.cryptensor 
    # Encrypt features / labels
    x = crypten.cryptensor(x)
    y = crypten.cryptensor(y)

    logging.info("==================")
    logging.info("CrypTen Training")
    logging.info("==================")
# NOTE：训练无差异
    begin_time = time.time()
    w, b = train_linear_svm(x, y, epochs=epochs, lr=lr, print_time=True)
    end_time = time.time()
    
    time_all = end_time - begin_time
    print("time:", time_all)
    # if not skip_plaintext:
    #     logging.info("PyTorch Weights  :")
    #     logging.info(w_torch)
    # logging.info("CrypTen Weights:")
# NOTE：get_plain_text()
    # logging.info(w.get_plain_text())

    # if not skip_plaintext:
    #     logging.info("PyTorch Bias  :")
    #     logging.info(b_torch)
    # logging.info("CrypTen Bias:")
    # logging.info(b.get_plain_text())
