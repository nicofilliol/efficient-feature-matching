import torch
from SuperGlue.models.matching import Matching
from SuperGlue.models.superglue import AttentionalPropagation
from quantization.linear_quantization import linear_quantize_weight, linear_quantize_weight_per_channel, plot_weight_distribution, fuse_conv_bn, linear_quantize_matching
from quantization.kmeans_quantization import KMeansQuantizer
from evaluate import evaluate
import helper
import os
from pathlib import Path
from SuperGlue.utils.common import read_image_with_homography, download_base_files, download_test_images
import numpy as np

recover_model = lambda model: model.load_state_dict("/Users/nfilliol/Desktop/ETH/MIT_HS22/TinyML/efficient-feature-matching/SuperGlue/models/weights/superglue_cocohomo.pt")

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

@torch.no_grad()
def peek_linear_quantization(model):
    for bitwidth in [4, 2]:
        for name, param in model.named_parameters():
            if param.dim() > 1:
                quantized_param, scale, zero_point = \
                    linear_quantize_weight_per_channel(param, bitwidth)
                param.copy_(quantized_param)
        plot_weight_distribution(model, bitwidth)
        recover_model(model)

def add_range_recoder_hook(model, input_activation, output_activation):
    import functools
    def _record_range(self, x, y, module_name):
        x = x[0]
        input_activation[module_name] = x.detach()
        output_activation[module_name] = y.detach()

    all_hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.ReLU, torch.nn.Conv1d)):
            all_hooks.append(m.register_forward_hook(
                functools.partial(_record_range, module_name=name)))
    return all_hooks


def fuse_batchnorm_layers(model):
    #  fuse the batchnorm into conv layers
    model_fused = model
    modules = model_fused.modules()
    for module in model_fused.modules():
        fused_mlp = []
        if isinstance(module, AttentionalPropagation):
            fused_mlp.append(fuse_conv_bn(
                module.mlp[0], module.mlp[1]))
            fused_mlp.append(module.mlp[2])
            fused_mlp.append(module.mlp[3])
            module.mlp = torch.nn.Sequential(*fused_mlp)


def quantize_inputs(x):
    quantized = (x * 255 - 128).clamp(-128, 127).to(torch.int8)
    return quantized

def main():
    # Evaluate dense model
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.0,
            'max_keypoints': 512,
            'remove_borders': 4
        },
        'superglue': {
            'weights_path': "/Users/nfilliol/Desktop/ETH/MIT_HS22/TinyML/efficient-feature-matching/SuperGlue/models/weights/superglue_cocohomo.pt",
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
            'num_layers': 9,
            'use_layernorm': False,
            'bin_value': 1.0,
            'pos_loss_weight': 0.45,
            'neg_loss_weight': 1.0,
            'restore_path': ""
        }
    }

    device = 'cpu'
    matching_dense = Matching(config).eval().to(device)
    matching_quantized = Matching(config).eval().to(device)
    fuse_batchnorm_layers(matching_quantized)

    # Should not change the model performance
    #helper.profile_matching_model(matching_dense)
    #helper.profile_matching_model(matching_quantized)

    #results_dense = evaluate(matching_dense)
    #results_quantized = evaluate(matching_quantized)

    # add hook to record the min max value of the activation
    input_activation = {}
    output_activation = {}
    hooks = add_range_recoder_hook(matching_quantized, input_activation, output_activation)
    inp0, inp1 = helper.get_sample_data()
    pred = matching_quantized({'image0': inp0, 'image1': inp1})


    # Remove hooks
    for h in hooks:
        h.remove()

    #print(input_activation, output_activation)
    linear_quantize_matching(matching_quantized, input_activation, output_activation)

    # Check model size
    print_model_size(matching_dense)
    print_model_size(matching_quantized)

    #results = evaluate(matching_dense)
    results_quantized = evaluate(matching_quantized, preprocess=quantize_inputs, max_evaluation_points=10)

    #plot_weight_distribution(model_quantized, 4)
    #results_quantized = evaluate(matching_quantized)
    #helper.profile_matching_model(matching_quantized)

def main_kmeans_quantization():
    # Evaluate dense model
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.0,
            'max_keypoints': 512,
            'remove_borders': 4
        },
        'superglue': {
            'weights_path': "/Users/nfilliol/Desktop/ETH/MIT_HS22/TinyML/efficient-feature-matching/SuperGlue/models/weights/superglue_cocohomo.pt",
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
            'num_layers': 9,
            'use_layernorm': False,
            'bin_value': 1.0,
            'pos_loss_weight': 0.45,
            'neg_loss_weight': 1.0,
            'restore_path': ""
        }
    }

    device = 'cpu'
    matching_dense = Matching(config).eval().to(device)
    matching_quantized = Matching(config).eval().to(device)

    # Apply K-Means quantization
    quantizer = KMeansQuantizer(matching_quantized.superpoint, 8, neglect_params=["convPa.weight", "convPb.weight", "convDa.weight", "convDb.weight"])

    # Check model size
    print_model_size(matching_dense)
    print_model_size(matching_quantized)

    #results = evaluate(matching_dense)
    results_quantized = evaluate(matching_quantized, preprocess=quantize_inputs, max_evaluation_points=10)



if __name__ == "__main__":
    #main_linear_quantization()
    main_kmeans_quantization()