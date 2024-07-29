import crypten.mpc
import pandas as pd
import numpy as np
import torch
import os
import random
import time
import csv
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
from crypten.mpc import MPCTensor
from crypten.mpc.primitives import BinarySharedTensor
import crypten.common
import crypten.common.functions as functions

crypten.init()


DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')

class DataSetParam:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

class DataSet:
    sonar = DataSetParam('sonar.csv')
    binary_dataset = DataSetParam("binary_dataset.csv")
    
    
    
def check_data_type(tensor):
    print(f"Type: {type(tensor)}")
    if isinstance(tensor, torch.Tensor):
        print(f"PyTorch Tensor dtype: {tensor.dtype}")
    elif isinstance(tensor, crypten.mpc.MPCTensor):
        print("This is a MPCTensor.")
    elif isinstance(tensor, crypten.mpc.primitives.binary.BinarySharedTensor):
        print("This is a BinarySharedTensor.")
    else:
        print("Unknown data type.")
    

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
    
    # 创建标签映射
    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_labels = np.array([label_to_int[label] for label in labels], dtype=np.uint8)
    
    # 将标签转换为独热矩阵
    num_classes =  len(unique_labels)
    one_hot_labels = np.zeros((int_labels.size, num_classes), dtype=np.uint8)
    one_hot_labels[np.arange(int_labels.size), int_labels] = 1

    # print("======feature_size=====", feature)
    # print("======label_size=====", one_hot_labels)
    return feature, one_hot_labels


# region subset partition
def setPartition(x):
    x_A = x.to(crypten.mpc.arithmetic)
    xx_T = x_A.matmul(x_A.t()) 
    xx_T_nor = xx_T.eq(0)   # a - Eq(a)，转换为1和-1
    xx_result = xx_T - xx_T_nor

    return xx_result
# endregion

# region ramp function
def ramp_function(x, max_feature, min_feature, prg1, prg2):
    # part 1
    cmp_result_1 = x > min_feature
    result_1 = cmp_result_1.to(crypten.mpc.arithmetic)
    
    # result 2
    temp = x > max_feature
    x_share_minus_one = x - prg1
    mux_result = crypten.where(temp, 0, x_share_minus_one)
    result_2 = mux_result + prg2
    
    N_ramp = result_1 * result_2
    # rank = comm.get().get_rank()
    # crypten.print(f"\nRank {rank}:\n result_1:{result_1.get_plain_text()}\n" \
    #             f" temp: {temp.get_plain_text()}\n",\
    #             f" mux_result: {mux_result.get_plain_text()}\n",\
    #             f" result_2:{result_2.get_plain_text()}\n",\
    #             f" N_ramp_result:{N_ramp.get_plain_text()}\n",
    #             # f" L_time:{L_time}\n",
    #             in_order=True)
    return N_ramp
# endregion
    

# region calculate distance
def calculateDis(feature, label, target_index, prg1, prg2):
    feature_target = feature[target_index, :].unsqueeze(0)
    label_target = label[target_index, :].unsqueeze(0)
    
    max_feature, max_feature_arg = functions.maximum.max(feature, dim=0)
    min_feature, min_feature_arg = functions.maximum.min(feature, dim=0)

    d = abs(feature_target - feature)
    d_judge = (d - min_feature).div(max_feature - min_feature)

    dis_ramp = ramp_function(d_judge, max_feature, min_feature, prg1, prg2)
    
    # rank = comm.get().get_rank()
    # crypten.print(f"Rank {rank}:\n d_judge: \n {d_judge.get_plain_text()}", in_order=True)
    
    return dis_ramp
# endregion


# region secret share
@mpc.run_multiprocess(world_size=2)
def secret_share(feature, label):
    feature_share = crypten.cryptensor(feature, ptype=crypten.mpc.arithmetic)  
    label_share =crypten.cryptensor(label, ptype = crypten.mpc.binary)  
    rank = comm.get().get_rank()
    crypten.print(f"Rank {rank}:\n {feature_share}", in_order=True)
    return feature_share, label_share
# endregion

@mpc.run_multiprocess(world_size=2)
def prg_share():
    one_share = MPCTensor(1, ptype = crypten.mpc.arithmetic)
    one_share_2 = MPCTensor(1, ptype = crypten.mpc.arithmetic)
    return one_share, one_share_2

# region relief
@mpc.run_multiprocess(world_size=2)
def relief(feature_share, label_share, random_index):
    subset_flag = setPartition(label_share)
    prg1, prg2 = prg_share()
    dis = calculateDis(feature_share, label_share, random_index, prg1, prg2)
    
    dis_sum = dis.sum(dim=1, keepdim=True)
    dis_max = functions.maximum.max(dis_sum)
    dis_temp = dis_max - dis_sum
    dis_map = subset_flag.matmul(dis_temp)
    
    
    # rank = comm.get().get_rank()
    # crypten.print(f"Rank {rank}:\n {label_share.get_plain_text()}", in_order=True)
# endregion
    
     
def main():
    data_train = DataSet.binary_dataset 
    feature, label = load_dataset(data_train.dataset_name)
    
    feature_share, label_share = secret_share(feature, label)
    
    t = 1  # 循环次数
    for _ in range(t):
        random_index = random.randint(0, feature.shape[0] - 1)
        dis_result = relief(feature_share, label_share, random_index)
        
    
main()





