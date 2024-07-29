import crypten
import crypten.mpc
import torch
import crypten.mpc as mpc
import crypten.communicator as comm 
import crypten.common as common
import random
from crypten.mpc import MPCTensor
from crypten.mpc.primitives import BinarySharedTensor, ArithmeticSharedTensor

crypten.init()

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

def secret_sharing(x,a,b):
    x_enc = crypten.cryptensor(x, ptype = crypten.mpc.arithmetic)
    a_enc = crypten.cryptensor(a, ptype = crypten.mpc.arithmetic)
    b_enc = crypten.cryptensor(b, ptype = crypten.mpc.arithmetic)
    
    return x_enc, a_enc, b_enc


@mpc.run_multiprocess(world_size=3) 
def intvlTest(x, a, b):
    x_enc, a_enc, b_enc = secret_sharing(x, a, b)
    cmp_a_x = x_enc.ge(a_enc)
    cmp_x_b = b_enc.ge(x_enc)
    cmp_a_x_share = cmp_a_x.share
    cmp_a_x_binary = BinarySharedTensor(cmp_a_x_share)
    check_data_type(cmp_a_x_binary)
    
    cmp_x_b_share = cmp_x_b.share 
    cmp_x_b_binary = BinarySharedTensor(cmp_x_b_share)
    check_data_type(cmp_x_b_binary)
    
    
    # crypten.print(f"\nRank {rank}:\n cmp_a_x: {cmp_a_x.get_plain_text()}\n", in_order=True)
    # crypten.print(f"\nRank {rank}:\n cmp_x_b: {cmp_x_b.get_plain_text()}\n", in_order=True)
    
    result_binary = cmp_a_x_binary & cmp_x_b_binary # binrysharedtensor
    # result_binary_share = result_binary.share
    # result_arith = crypten.mpc.MPCTensor(result_binary_share, ptype = crypten.mpc.binary)  
    rank = comm.get().get_rank()
    # result_biary = ArithmeticSharedTensor.convert(result_arith, ptype = crypten.ptype.binary)
    # crypten.print(f"\nRank {rank}:\n result: {result_arith}\n", in_order=True)
    crypten.print(f"\nRank {rank}:\n result: {result_binary}\n", in_order=True)



def main():
    # 区间范围
    a = 0
    b = 1
    
    #随机数范围
    x = random.randint(-10, 10) 
    print("x", x)
    intvlTest(x,a,b)
    
main()