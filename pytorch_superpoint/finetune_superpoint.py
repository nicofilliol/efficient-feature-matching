from SuperGlue.matching.models.superpoint import SuperPoint
from train4 import train_joint
import torch
import argparse
import yaml
import os
from settings import EXPER_PATH

def finetune_superpoint(model: SuperPoint, pruners: list, config='configs/superpoint_coco_train_heatmap.yaml'):
    torch.set_default_tensor_type(torch.FloatTensor)
    args = argparse.Namespace(command='train_joint', config=config, debug=True, eval=True, exper_name='superpoint_coco', func=train_joint)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(EXPER_PATH, args.exper_name)
    os.makedirs(output_dir, exist_ok=True)
    train_joint(config, output_dir, args, model=model)