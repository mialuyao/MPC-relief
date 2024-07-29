import torch

def generate_boolean_tensor(N):
    # 生成一个长度为N的随机布尔tensor
    random_tensor = torch.randint(0, 2, (N,), dtype=torch.bool)
    return random_tensor

# 示例
N = 10
bool_tensor = generate_boolean_tensor(N)
print(bool_tensor)
