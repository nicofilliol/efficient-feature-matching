import helper
from SuperGlue.models.matching import Matching
import random

def main():
    config_dense = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights_path': "/Users/nfilliol/Desktop/ETH/MIT_HS22/TinyML/efficient-feature-matching/SuperGlue/models/weights/superglue_cocohomo.pt",
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    config_pruned = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1
        },
        'superglue': {
            'weights_path': "/Users/nfilliol/Desktop/ETH/MIT_HS22/TinyML/efficient-feature-matching/SuperGlue/models/weights/superglue_finegrained_finetuned.pt",
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    device = 'cpu'
    matching_dense = Matching(config_dense).eval().to(device)
    matching_pruned = Matching(config_pruned).eval().to(device)

    # Profile Model
    param_sets_dense = [(name, param) for (name, param) in matching_dense.named_parameters() if param.dim() > 1 and "superglue" in name]
    sample_param_sets_dense = random.sample(param_sets_dense, 16)
    sample_param_sets_pruned = [(name, param) for (name, param) in matching_pruned.named_parameters() if name in dict(sample_param_sets_dense)]
    helper.plot_weight_distribution(sample_param_sets_dense, out_path="images/weights_superglue_dense.png")
    helper.plot_weight_distribution(sample_param_sets_pruned, out_path="images/weights_superglue_finetuned.png", count_nonzero_only=True)
   
    print("                ###### Dense Model ######")
    helper.profile_matching_model(matching_dense)
    print("              ###### Finetuned Model ######")
    helper.profile_matching_model(matching_pruned, count_nonzero_only=True)


if __name__ == "__main__":
    main()