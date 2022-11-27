from SuperGlue.match_homography import match_homography
from SuperGlue.models.matching import Matching
from pruning.channel_pruning import ChannelPruner
import argparse
import torch

def main(opt):
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

    # Apply Channel Pruner 
    pruner = ChannelPruner(matching.superpoint, prune_ratio=0.2)

    matching.superpoint.load_state_dict(torch.load("/Users/nfilliol/Desktop/ETH/MIT_HS22/TinyML/Project/Experiments/Channel_Finegrained/superpoint_finetuned-2.pt"))
    matching.superglue.load_state_dict(torch.load("/Users/nfilliol/Desktop/ETH/MIT_HS22/TinyML/Project/Experiments/Channel_Finegrained/superglue_finetuned-2.pt"))

    match_homography(opt, model=matching)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    opt = argparse.Namespace(input_homography='SuperGlue/assets/coco_test_images_homo.txt', input_dir='SuperGlue/assets/coco_test_images/', output_dir='output/match_visualization/', max_length=-1, resize=[640, 480], resize_float=False, superglue='coco_homo', max_keypoints=1024, keypoint_threshold=0.005, nms_radius=4, sinkhorn_iterations=20, min_matches=12, match_threshold=0.2, viz=True, eval=True, fast_viz=False, cache=False, show_keypoints=True, viz_extension='png', opencv_display=False, shuffle=False, force_cpu=False)
    print(opt)
    main(opt)