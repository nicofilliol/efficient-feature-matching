from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import cv2
import os
from SuperGlue.models.matching import Matching
from SuperGlue.utils.common import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,read_image_with_homography,
                          rotate_intrinsics, rotate_pose_inplane, compute_pixel_error,
                          scale_intrinsics, weights_mapping, download_base_files, download_test_images)
from SuperGlue.utils.preprocess_utils import torch_find_matches

download_base_files()
download_test_images()

@torch.no_grad()
def evaluate(model: Matching, max_evaluation_points=-1):

    config = {
        "input_homography" : "SuperGlue/assets/coco_test_images_homo.txt",
        "input_dir" : "SuperGlue/assets/coco_test_images/",
        "output_dir" : "dump_homo_pairs/",
        "max_evaluation_points" : max_evaluation_points,
        "shuffle" : True,
        "resize" : [640, 480],
        "resize_float" : True,
        "min_matches" : 12,
        **model.superglue.config,
        **model.superpoint.config
    }

    with open(config["input_homography"], 'r') as f:
        homo_pairs = f.readlines()

    if config["max_evaluation_points"] > -1:
        homo_pairs = homo_pairs[0:np.min([len(homo_pairs), config["max_evaluation_points"]])]

    if config["shuffle"]:
        random.Random(0).shuffle(homo_pairs)
    
    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matching = model.to(device)

    print('Running inference on device \"{}\"'.format(device))

    # Setup input data
    input_dir = Path(config["input_dir"])
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    print('Will write evaluation results to directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)
    for i, info in enumerate(homo_pairs):
        split_info = info.strip().split(' ')
        image_name = split_info[0]
        homo_info = list(map(lambda x: float(x), split_info[1:]))
        homo_matrix = np.array(homo_info).reshape((3,3)).astype(np.float32)
        stem0 = Path(image_name).stem
        matches_path = output_dir / '{}_matches.npz'.format(stem0)
        eval_path = output_dir / '{}_evaluation.npz'.format(stem0)        

        # Handle --cache logic.
        do_match = True
        do_eval = True

        if not (do_match or do_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(homo_pairs)))
            continue
        image0, image1, inp0, inp1, scales0, homo_matrix = read_image_with_homography(input_dir / image_name, homo_matrix, device,
                                                config["resize"], 0, config["resize_float"])

        if image0 is None or image1 is None:
            print('Problem reading image pair: {}'.format(
                input_dir/ image_name))
            exit(1)
        timer.update('load_image')

        if do_match:
            # Perform the matching.
            pred = matching({'image0': inp0, 'image1': inp1})
            kp0_torch, kp1_torch = pred['keypoints0'][0], pred['keypoints1'][0]
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        ma_0, ma_1, miss_0, miss_1 = torch_find_matches(kp0_torch, kp1_torch, torch.from_numpy(homo_matrix).to(kp0_torch.device), dist_thresh=3, n_iters=3)
        ma_0, ma_1 = ma_0.cpu().numpy(), ma_1.cpu().numpy()
        gt_match_vec = np.ones((len(matches), ), dtype=np.int32) * -1
        gt_match_vec[ma_0] = ma_1
        corner_points = np.array([[0,0], [0, image0.shape[0]], [image0.shape[1], image0.shape[0]], [image0.shape[1], 0]]).astype(np.float32)
        if do_eval:
            if len(mconf) < config["min_matches"]:
                out_eval = {'error_dlt': -1,
                            'error_ransac': -1,
                            'precision': -1,
                            'recall': -1
                            }
                #non matched points will not be considered for evaluation
                np.savez(str(eval_path), **out_eval)
                timer.update('eval')
                print('Skipping {} due to inefficient matches'.format(i))
                continue
            sort_index = np.argsort(mconf)[::-1][0:4]
            est_homo_dlt = cv2.getPerspectiveTransform(mkpts0[sort_index, :], mkpts1[sort_index, :])
            est_homo_ransac, _ = cv2.findHomography(mkpts0, mkpts1, method=cv2.RANSAC, maxIters=3000)
            corner_points_dlt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_dlt).squeeze(1)
            corner_points_ransac = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), est_homo_ransac).squeeze(1)
            corner_points_gt = cv2.perspectiveTransform(np.reshape(corner_points, (-1, 1, 2)), homo_matrix).squeeze(1)
            error_dlt = compute_pixel_error(corner_points_dlt, corner_points_gt)
            error_ransac = compute_pixel_error(corner_points_ransac, corner_points_gt)
            match_flag = (matches[ma_0] == ma_1)
            precision = match_flag.sum() / valid.sum()
            fn_flag = np.logical_and((matches != gt_match_vec), (matches == -1))
            recall = match_flag.sum() / (match_flag.sum() + fn_flag.sum())
            # Write the evaluation results to disk.
            out_eval = {'error_dlt': error_dlt,
                        'error_ransac': error_ransac,
                        'precision': precision,
                        'recall': recall
                        }
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(homo_pairs)))

    # Collate the results into a final table and print to terminal.
    errors_dlt = []
    errors_ransac = []
    precisions = []
    recall = []
    matching_scores = []
    for info in homo_pairs:
        split_info = info.strip().split(' ')
        image_name = split_info[0]
        stem0 = Path(image_name).stem
        eval_path = output_dir / '{}_evaluation.npz'.format(stem0)
        results = np.load(eval_path)
        if results['precision'] == -1:
            continue
        errors_dlt.append(results['error_dlt'])
        errors_ransac.append(results['error_ransac'])
        precisions.append(results['precision'])
        recall.append(results['recall'])
    thresholds = [5, 10, 25]
    aucs_dlt = pose_auc(errors_dlt, thresholds)
    aucs_ransac = pose_auc(errors_ransac, thresholds)
    aucs_dlt = [100.*yy for yy in aucs_dlt]
    aucs_ransac = [100.*yy for yy in aucs_ransac]
    prec = 100.*np.mean(precisions)
    rec = 100.*np.mean(recall)
    print('Evaluation Results (mean over {} pairs):'.format(len(homo_pairs)))
    print("For DLT results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs_dlt[0], aucs_dlt[1], aucs_dlt[2], prec, rec))
    print("For homography results...")
    print('AUC@5\t AUC@10\t AUC@25\t Prec\t Recall\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs_ransac[0], aucs_ransac[1], aucs_ransac[2], prec, rec))

    results = {
        'precision': prec,
        'recall': rec,
        'dlt' : {
            'auc5': aucs_dlt[0],
            'auc10': aucs_dlt[1],
            'auc25': aucs_dlt[2],
        },
        'ransac': {
            'auc5': aucs_ransac[0],
            'auc10': aucs_ransac[1],
            'auc25': aucs_ransac[2],
        }
    }

    model.to('cpu')
    return results



if __name__ == "__main__":
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
    evaluate(matching)