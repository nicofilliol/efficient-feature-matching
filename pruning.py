import torch
import numpy as np
import copy
from tqdm.auto import tqdm
from pruning.fine_grained_pruning import FineGrainedPruner
from evaluate import evaluate
from finetune import finetune
from SuperGlue.models.matching import Matching
import matplotlib.pyplot as plt
import math
import helper
import random

def check_sparsity(model: torch.nn.Module, param_name, sparsity):
    for (name, param) in model.named_parameters():
        if param.dim() > 1:
            sp = helper.get_sparsity(param) 
            print(f"Layer: {name}, sparsity: {sp:.2f}")
            if name == param_name:
                assert abs(sp - sparsity) < 0.01, f"Param has sparsity {sp}, expected: {sparsity}"
            else:
                assert sp < 0.1, f"Param has unexpectedly high sparsity."
        
    print("Sparsity check successful.")

@torch.no_grad()
def sensitivity_scan(model, named_params, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    precisions = []
    recalls = []
    names = []
    for i_layer, (name, param) in enumerate(named_params):
        model_clone = copy.deepcopy(model)
        precision = []
        recall = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_params)} weight - {name}'):
            sparsity_dict = { name : sparsity}
            pruned = FineGrainedPruner(model_clone, sparsity_dict)

            check_sparsity(pruned.model, name, sparsity)

            results = evaluate(pruned.model, max_evaluation_points=-1)
            prec, rec = results["precision"], results["recall"]
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: precision={prec:.2f}, recall={prec:.2f}', end='')
            precision.append(prec)
            recall.append(rec)
        if verbose:
            print(f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: precision=[{", ".join(["{:.2f}".format(x) for x in precision])}]: recall=[{", ".join(["{:.2f}".format(x) for x in recall])}]', end='')
        precisions.append(precision)
        recalls.append(recall)
        names.append(name)
    return (names, sparsities, precisions, recalls)

def plot_sensitivity_scan(names, sparsities, results, dense_model_result, metric="precision"):
    lower_bound = 100 - (100 - dense_model_result) * 1.5
    fig, axes = plt.subplots(3, int(math.ceil(len(results) / 3)),figsize=(15,8))
    axes = axes.ravel()
    plot_index = 0
    for name in names:
        ax = axes[plot_index]
        curve = ax.plot(sparsities, results[plot_index])
        line = ax.plot(sparsities, [lower_bound] * len(sparsities))
        ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
        ax.set_ylim(lower_bound-10, 95)
        ax.set_title(name)
        ax.set_xlabel('sparsity')
        ax.set_ylabel(metric)
        ax.legend([
            f'{metric} after pruning',
            f'{lower_bound / dense_model_result * 100:.0f}% of dense model {metric}'
        ])
        ax.grid(axis='x')
        plot_index += 1
    fig.suptitle(f'Sensitivity Curves: Validation {metric} vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.savefig(f"{metric}_sensitivity.png")

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
    matching = Matching(config).eval().to(device)

    # # Randomly sample a subset of parameter sets for tests
    # param_sets = [(name, param) for (name, param) in matching.named_parameters() if param.dim() > 1 and "superglue" in name]
    # sample_param_sets = random.sample(param_sets, 12)

    # # Profile and evaluate dense model
    # helper.profile_matching_model(matching, count_nonzero_only=False)
    # helper.plot_weight_distribution(sample_param_sets, out_path="weight_distribution_dense.png", count_nonzero_only=False)

    # dense_results = evaluate(matching)
    # print(f"Dense Results = {dense_results}")
    
    # # Print Model Structure and plot distribution of weights
    # helper.print_model_structure(matching.superglue)

    # # Fine-grained Pruning (magnitude based)
    # # Sensitivity Scan
    # names, sparsities, precisions, recalls = sensitivity_scan(matching, sample_param_sets, scan_step=0.2, scan_start=0.5, scan_end=1.0)
    # plot_sensitivity_scan(names, sparsities, precisions, dense_results["precision"], metric="precision")
    # plot_sensitivity_scan(names, sparsities, recalls, dense_results["recall"], metric="recall")

    # Prune
    sparsity_for_layer_type = {
        'gnn_attn' : 0.9,
        'mlp' : 0.7,
    }

    sparsity_dict = {}
    for name, _ in matching.named_parameters():
        if "superglue" in name:
            if "attn" in name:
                sparsity_dict[name] = sparsity_for_layer_type["gnn_attn"]
            elif "mlp" in name:
                sparsity_dict[name] = sparsity_for_layer_type["mlp"]
            else:
                sparsity_dict[name] = 0

    pruner = FineGrainedPruner(matching, sparsity_dict)
    # helper.profile_matching_model(pruner.model, count_nonzero_only=True)
    # helper.plot_weight_distribution(sample_param_sets, out_path="weight_distribution_pruned.png", count_nonzero_only=True)
    # pruned_results = evaluate(pruner.model, max_evaluation_points=-1)
    # print(f"Pruned Results = {pruned_results}")

    finetune(pruner.model, max_epochs=1)


if __name__ == "__main__":
    main()