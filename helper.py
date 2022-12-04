import torchprofile
import torch
from torch import nn
import matplotlib.pyplot as plt
import time
import warnings
import math
from pathlib import Path
from SuperGlue.utils.common import read_image_with_homography, download_base_files, download_test_images
import numpy as np
import timeit

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

def get_sample_data(idx=0):
    with open("SuperGlue/assets/coco_test_images_homo.txt", 'r') as f:
        homo_pairs = f.readlines()
    
    homo_pair = homo_pairs[idx]
    download_base_files()
    download_test_images()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    split_info = homo_pair.strip().split(' ')
    image_name = split_info[0]
    homo_info = list(map(lambda x: float(x), split_info[1:]))
    homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)

    input_dir = Path("SuperGlue/assets/coco_test_images/")
    image0, image1, inp0, inp1, scales0, homo_matrix = read_image_with_homography(input_dir / image_name, homo_matrix, device,
                                                [640, 480], 0, False)
    return inp0, inp1


class MatchingProfile():
    def __init__(self, matching, keys=['keypoints', 'scores', 'descriptors'], n_warmup=20, n_test=100, count_nonzero_only=False):
        self.matching = matching
        self.keys = keys
        self.n_warmup = n_warmup
        self.n_test = n_test
        self.count_nonzero_only = count_nonzero_only

    @torch.no_grad()
    def profile_superpoint(self):
        dummy_input = {"image" : get_sample_data()[0]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs = torchprofile.profile_macs(self.matching.superpoint, dummy_input)
        params = get_num_parameters(self.matching.superpoint, count_nonzero_only=self.count_nonzero_only)
        model_size = get_model_size(self.matching.superpoint, count_nonzero_only=self.count_nonzero_only)

        # Warmup
        for i in range(self.n_warmup):
            # Get dummy input
            input = {"image" : get_sample_data(idx=i)[0]}
            out = self.matching.superpoint(input)

        n_keypoints = len(out["keypoints"][0])
        # Real test
        n_images = 100
        input = [{"image" : get_sample_data(idx=i)[0]} for i in range(n_images)]

        latency_dict = {}
        overall_avg = 0
        n_images = 100

        for n in range(n_images):
            t1 = time.time()
            for i in range(self.n_test):
                out = self.matching.superpoint(input[n])
            t2 = time.time()
            avg_latency =(t2-t1) / self.n_test
            overall_avg += avg_latency/n_images
            n_keypoints = len(out["keypoints"][0])
            if n_keypoints in latency_dict:
                latency_dict[n_keypoints] = np.mean([latency_dict[n_keypoints], avg_latency])
            else:
                latency_dict[n_keypoints] = avg_latency

        return (overall_avg, macs, params, model_size, n_keypoints, latency_dict)

    @torch.no_grad()
    def profile_superglue(self):
        # Create dummy input
        inp0, inp1 = get_sample_data(idx=0)
        data = {'image0': inp0, 'image1': inp1}

        pred0 = self.matching.superpoint({'image': data['image0']})
        data = {**data, **{k+'0': v for k, v in pred0.items()}}
        pred1 = self.matching.superpoint({'image': data['image1']})
        data = {**data, **{k+'1': v for k, v in pred1.items()}}

        n_keypoints0 = len(pred0["keypoints"][0])
        n_keypoints1 = len(pred1["keypoints"][0])
        
        def batch_data(data):
            # Batch all data
            for k in data:
                if isinstance(data[k], (list, tuple)):
                    data[k] = torch.stack(data[k])
            return data

        data = batch_data(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            macs = torchprofile.profile_macs(self.matching.superglue, data)
        params = get_num_parameters(self.matching.superglue, count_nonzero_only=self.count_nonzero_only)
        model_size = get_model_size(self.matching.superglue, count_nonzero_only=self.count_nonzero_only)

        # Perform the matching
        # Warmup
        for _ in range(self.n_warmup):
            out = {**data, **self.matching.superglue(data)}
        
        kpts0, kpts1 = out['keypoints0'], out['keypoints1']
        matches, conf = out['matches0'], out['matching_scores0']
        valid = matches > -1
        mkpts0 = kpts0[valid]
        n_matches = len(mkpts0)
        
        # Real test
        latency_dict = {}
        overall_avg = 0
        n_images = 100
        for n in range(n_images):
            inp0, inp1 = get_sample_data(idx=n)
            data = {'image0': inp0, 'image1': inp1}
            pred0 = self.matching.superpoint({'image': data['image0']})
            data = {**data, **{k+'0': v for k, v in pred0.items()}}
            pred1 = self.matching.superpoint({'image': data['image1']})
            data = {**data, **{k+'1': v for k, v in pred1.items()}}
            data = batch_data(data)

            t1 = time.time()
            for _ in range(self.n_test):
                _ = {**data, **self.matching.superglue(data)}
            t2 = time.time()
            avg_latency = (t2 - t1) / self.n_test
            overall_avg += avg_latency/n_images
            n_keypoints = len(pred0["keypoints"][0])
            if n_keypoints in latency_dict:
                latency_dict[n_keypoints] = np.mean([latency_dict[n_keypoints], avg_latency])
            else:
                latency_dict[n_keypoints] = avg_latency

        return (overall_avg, macs, params, model_size, n_keypoints0, n_keypoints1, n_matches, latency_dict)

    def to_cpu(self):
        self.matching = self.matching.to('cpu')
    
    def to_gpu(self):
        self.matching = self.matching.to('cuda')


def profile_matching_model(model, count_nonzero_only=False):
    matching_latency = MatchingProfile(model, n_warmup=5, n_test=10, count_nonzero_only=count_nonzero_only)

    # CPU Latency Measurement
    table_template = "{:<15} {:<15} {:<15}"
    print (table_template.format('', 'SuperPoint', 'SuperGlue'))
    matching_latency.to_cpu()

    sp_latency, sp_macs, sp_params, sp_size, n_keypoints, sp_latencies = matching_latency.profile_superpoint()
    sg_latency, sg_macs, sg_params, sg_size, n_keypoints0, n_keypoints1, n_matches, sg_latencies = matching_latency.profile_superglue()
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

    print(f"SuperPoint: {n_keypoints} keypoints.")
    print(f"SuperGlue: {n_keypoints0} keypoints in image A, {n_keypoints1} in image B, {n_matches} matches.")

    print(sp_latencies)
    print(sg_latencies)

    # Put model back to CUDA
    #matching_latency.to_gpu()