import helper
from SuperGlue.models.matching import Matching
import random

def main():
    config = {
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

    device = 'cpu'
    matching = Matching(config).eval().to(device)

    # Profile Model
    dense_model_size = helper.get_model_size(matching.superglue)
    print(f"Dense model has size={dense_model_size/helper.MiB:.2f} MiB")

    param_sets = [(name, param) for (name, param) in matching.named_parameters() if param.dim() > 1 and "superglue" in name]
    sample_param_sets = random.sample(param_sets, 16)
    helper.plot_weight_distribution(sample_param_sets, out_path="weight_distribution_superglue.png")
    helper.profile_matching_model(matching)


if __name__ == "__main__":
    main()