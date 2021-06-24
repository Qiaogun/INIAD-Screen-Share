import torch
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
import json
import pandas as pd

root_dir = os.path.abspath('./data_all/data/diff/')

X_trainS1 = pd.read_csv(os.path.join(root_dir, "Chinese_seven_diff.csv"))
X_trainS1.head()