import torch
import torch.nn as nn

# TODO 将每种网络的 下采样、特征处理、上采样、特征处理 import

class Network(nn.Module):
    def __init__(self, model_name, input_dim, output_dim, cfg) -> None:
        super().__init__()
        self.input_dim = input_dim