
import random
import numpy as np

import torch


def fix_random_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
