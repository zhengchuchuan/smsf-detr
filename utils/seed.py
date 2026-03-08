
# 固定训练的随机种子
def fixed_random_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # 设置为确定性算法
    torch.backends.cudnn.benchmark = False # 关闭自动寻找最优算法