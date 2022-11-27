import torchprofile
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import warnings
import math


# Data Size Definitions
Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero().item()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

def get_number_of_paremeter_sets(model: nn.Module):
    count = 0
    for name, param in model.named_parameters():
        count += 1
    return count

def print_model_structure(model):
    for name, param in model.named_parameters():
        if param.dim() > 1:
            print(f"{name}: # elements = {torch.numel(param)}, sparsity = {get_sparsity(param)}")

def plot_weight_distribution(layers, out_path, bins=256, count_nonzero_only=False):
    fig, axes = plt.subplots(4, int(math.ceil(len(layers) / 4)),figsize=(15,15))
    axes = axes.ravel()
    plot_index = 0
    for name, param in layers:
        if param.dim() > 1:
            ax = axes[plot_index]
            if count_nonzero_only:
                param_cpu = param.detach().reshape(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].reshape(-1)
                ax.hist(param_cpu, bins=bins, density=True, 
                        color = 'blue', alpha = 0.5)
            else:
                ax.hist(param.detach().reshape(-1).cpu(), bins=bins, density=True, 
                        color = 'blue', alpha = 0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
            if plot_index >= len(layers):
                break

    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(out_path)


class MatchingProfile():
    def __init__(self, matching, keys=['keypoints', 'scores', 'descriptors'], n_warmup=20, n_test=100, count_nonzero_only=False):
        self.matching = matching
        self.keys = keys
        self.n_warmup = 20
        self.n_test = 100
        self.count_nonzero_only = count_nonzero_only

    @torch.no_grad()
    def profile_superpoint(self):
        # Create dummy input
        dummy_input = {'image': torch.randn(1, 1, 480, 640)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs = torchprofile.profile_macs(self.matching.superpoint, dummy_input)
        params = get_num_parameters(self.matching.superpoint, count_nonzero_only=self.count_nonzero_only)
        model_size = get_model_size(self.matching.superpoint, count_nonzero_only=self.count_nonzero_only)

        # Warmup
        for _ in range(self.n_warmup):
            _ = self.matching.superpoint(dummy_input)
        # Real test
        t1 = time.time()
        for _ in range(self.n_test):
            _ = self.matching.superpoint(dummy_input)
        t2 = time.time()
        avg_latency = (t2 - t1) / self.n_test

        return (avg_latency, macs, params, model_size)

    @torch.no_grad()
    def profile_superglue(self):
        # Create dummy input
        data = {'image0': torch.randn(1, 1, 480, 640), 'image1': torch.randn(1, 1, 480, 640)}

        pred0 = self.matching.superpoint({'image': data['image0']})
        data = {**data, **{k+'0': v for k, v in pred0.items()}}
        pred1 = self.matching.superpoint({'image': data['image1']})
        data = {**data, **{k+'1': v for k, v in pred1.items()}}
        
        # Batch all data
        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs = torchprofile.profile_macs(self.matching.superglue, data)
        params = get_num_parameters(self.matching.superglue, count_nonzero_only=self.count_nonzero_only)
        model_size = get_model_size(self.matching.superglue, count_nonzero_only=self.count_nonzero_only)

        # Perform the matching
        # Warmup
        for _ in range(self.n_warmup):
            _ = {**data, **self.matching.superglue(data)}
        # Real test
        t1 = time.time()
        for _ in range(self.n_test):
            _ = {**data, **self.matching.superglue(data)}
        t2 = time.time()
        avg_latency = (t2 - t1) / self.n_test

        return (avg_latency, macs, params, model_size)

    def to_cpu(self):
        self.matching = self.matching.to('cpu')
    
    def to_gpu(self):
        self.matching = self.matching.to('cuda')


def profile_matching_model(model, count_nonzero_only=False):
    matching_latency = MatchingProfile(model, count_nonzero_only=count_nonzero_only)

    # CPU Latency Measurement
    table_template = "{:<15} {:<15} {:<15}"
    print (table_template.format('', 'SuperPoint', 'SuperGlue'))
    matching_latency.to_cpu()

    sp_latency, sp_macs, sp_params, sp_size = matching_latency.profile_superpoint()
    sg_latency, sg_macs, sg_params, sg_size = matching_latency.profile_superglue()
    print(table_template.format('Latency (ms)', 
                                round(sp_latency * 1000, 1),
                                round(sg_latency * 1000, 1)))
    print(table_template.format('MACs (M)', 
                                round(sp_macs / 1e6),
                                round(sg_macs / 1e6)))
    print(table_template.format('Param (M)', 
                                round(sp_params / 1e6, 2),
                                round(sg_params / 1e6, 2)))
    print(table_template.format('Size (MiB)', 
                                round(sp_size / MiB, 2),
                                round(sg_size / MiB, 2)))

    # Put model back to CUDA
    #matching_latency.to_gpu()