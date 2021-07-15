import datasets
from datasets.load import load_metric, load_dataset, load_dataset_builder
import torch
import torch.nn
import torch.nn.functional as F

dataset = load_dataset('load_dataset.py')
print(dataset)