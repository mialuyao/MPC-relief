
# import crypten
# import crypten.mpc
# import torch
# import crypten.mpc as mpc
# import crypten.mpc.primitives.beaver as beaver
# import crypten.communicator as comm 
# import torch
# from crypten.mpc import MPCTensor
# from crypten.mpc.primitives import BinarySharedTensor
# import time
# import random
# import crypten.common.functions

# crypten.init()


# # region 
# # DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

# # class DataSetParam:
# #     def __init__(self, dataset_name):
# #         self.dataset_name = dataset_name


# # class DataSet:
# #     sonar = DataSetParam('sonar.csv')
# #     sonar_selected = DataSetParam('sonar_selected.csv')
    
    
    
# # def load_dataset(dataset):
# #     dataset_path = os.path.join(DATASET_DIR, dataset)

# #     # 尝试打开并读取CSV文件
# #     try:
# #         with open(dataset_path, 'r', encoding='utf-8') as csvfile:
# #             spamreader = csv.reader(csvfile)
# #             data = np.array(list(spamreader))
# #     except FileNotFoundError:
# #         print(f"Error: The file {dataset_path} was not found.")
# #         return None, None
# #     except Exception as e:
# #         print(f"Error: An error occurred while reading the file: {e}")
# #         return None, None

# #     # 检查数据是否为空
# #     if data.size == 0:
# #         print("Error: The dataset is empty.")
# #         return None, None

# #     feature = data[:, :-1].astype(np.float64)
# #     labels = data[:, -1]

# #     # 创建标签映射
# #     unique_labels = np.unique(labels)
# #     label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
# #     int_labels = np.array([label_to_int[label] for label in labels], dtype=np.uint8)
# #     labell = int_labels.reshape(-1,1)

# #     # 将标签转换为独热矩阵
# #     one_hot_labels = np.zeros((int_labels.size, int_labels.max() + 1), dtype=np.uint8)
# #     one_hot_labels[np.arange(int_labels.size), int_labels] = 1

# #     # print("======feature_size=====", feature.shape)
# #     print("======label_size=====", one_hot_labels)
# #     return feature, one_hot_labels

# # data_train = DataSet.sonar

# # load_dataset(data_train.dataset_name)
# # endregion


# def generate_random():
#     random_tensor = random.uniform(-3, -2)
#     return random_tensor

# # region 
# # @mpc.run_multiprocess(world_size=2)
# # def test(x,y):
#     # x_enc = crypten.cryptensor(x, ptype=crypten.mpc.arithmetic)
#     # y_enc = crypten.cryptensor(y, ptype=crypten.mpc.arithmetic)
#     # ge_time_begin = time.time()
#     # ge_  = x_enc.ge(y_enc)
#     # ge_time_end = time.time()
#     # ge_time = ge_time_end - ge_time_begin
#     # print("ge_time", ge_time)
    
#     # gt_time_begin = time.time()
#     # gt_  = x_enc.gt(y_enc)
#     # gt_time_end = time.time()
#     # gt_time = gt_time_end - gt_time_begin
#     # print("gt_time", gt_time)
    
#     # relu_time_begin = time.time()
#     # relu_ = ge_.relu()
#     # relu_time_end = time.time()
#     # relu_time = relu_time_end - relu_time_begin
#     # print("relu_time", relu_time)
# # endregion


# # region N-ramp function
# @mpc.run_multiprocess(world_size=3)
# def N_ramp(x, min_value, max_value):
#     x_share = crypten.cryptensor(x, ptype = crypten.mpc.arithmetic)
#     min_value_share = crypten.cryptensor(min_value, ptype = crypten.mpc.arithmetic)
#     max_value_share = crypten.cryptensor(max_value, ptype = crypten.mpc.arithmetic)
#     one_share = crypten.cryptensor(1, ptype = crypten.mpc.arithmetic)
#     one_share_2 = crypten.cryptensor(1, ptype = crypten.mpc.arithmetic)
    
#     # part 1
#     cmp_result_1 = x_share > min_value_share
#     result_1 = cmp_result_1.to(crypten.mpc.arithmetic)
    
#     # result 2
#     temp = x_share > max_value_share
#     x_share_minus_one = x_share - one_share
#     mux_result = crypten.where(temp, 0, x_share_minus_one)
#     result_2 = mux_result + one_share_2
    
#     N_ramp = result_1 * result_2
#     rank = comm.get().get_rank()
#     crypten.print(f"\nRank {rank}:\n result_1:{result_1.get_plain_text()}\n" \
#                 f" temp: {temp.get_plain_text()}\n",\
#                 f" mux_result: {mux_result.get_plain_text()}\n",\
#                 f" result_2:{result_2.get_plain_text()}\n",\
#                 f" N_ramp_result:{N_ramp.get_plain_text()}\n",
#                 # f" L_time:{L_time}\n",
#                 in_order=True)

# def main():
#     x = generate_random()
#     print("x", x)
#     N_ramp(x, -2, 2)

# main()

# # endregion






import os
import numpy as np
import csv

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset/')

class DataSetParam:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

class DataSet:
    sonar = DataSetParam('sonar.csv')
    binary_dataset = DataSetParam("binary_dataset.csv")
    
def load_dataset(dataset):
    dataset_path = os.path.join(DATASET_DIR, dataset)
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

    if data.size == 0:
        print("Error: The dataset is empty.")
        return None, None

    feature = data[:, :-1].astype(np.float64)
    labels = data[:, -1]
    print("label", labels)
    # 创建标签映射
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels], dtype=np.uint8)
    print("int_labels", int_labels)
    # 将标签转换为独热矩阵
    num_classes =  len(unique_labels)
    one_hot_labels = np.zeros((int_labels.size, num_classes), dtype=np.uint8)
    one_hot_labels[np.arange(int_labels.size), int_labels] = 1
    print("one_hot_labels", one_hot_labels)

    return feature, one_hot_labels

data_train = DataSet.binary_dataset 
feature, label = load_dataset(data_train.dataset_name)
print("labelssssss", label)