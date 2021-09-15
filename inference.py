import os
import os.path as osp
from datetime import date
from pathlib import Path
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .util import misc
from .datasets import build_image_dataset
from .models import build_model
from .config import cfg

filepath = str(Path(__file__).resolve().parent)
print("filepath {}".format(filepath))
conf_file = filepath + '/config-files/ctrl-c.yaml'
checkpoint_file = filepath + '/logs/checkpoint.pth'
cfg.merge_from_file(conf_file)
checkpoint = torch.load(checkpoint_file, map_location='cpu')
device = torch.device(cfg.DEVICE)

model, _ = build_model(cfg)
model.to(device)

model.load_state_dict(checkpoint['model'])
model = model.eval()

def run_inference(sample_path):
    dataset_test = build_image_dataset(image_set=sample_path, cfg=cfg)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                 drop_last=False,
                                 collate_fn=misc.collate_fn,
                                 num_workers=1)

    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)

            zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            hl = outputs['pred_hl'].to('cpu')[0].numpy()
            fovy = fovy[0]
            up_vector = compute_up_vector(zvp, fovy)
            pitch, roll = decompose_up_vector(up_vector)
            return pitch.astype(float), roll.astype(float), fovy.astype(float)


def extract_hl(left, right, width):
    hl_homo = np.cross(np.append(left, 1), np.append(right, 1))
    hl_left_homo = np.cross(hl_homo, [-1, 0, -width/2]);
    hl_left = hl_left_homo[0:2]/hl_left_homo[-1];
    hl_right_homo = np.cross(hl_homo, [-1, 0, width/2]);
    hl_right = hl_right_homo[0:2]/hl_right_homo[-1];
    return hl_left, hl_right


def compute_horizon(hl, crop_sz, org_sz, eps=1e-6):
    a,b,c = hl
    if b < 0:
        a, b, c = -hl
    b = np.maximum(b, eps)
    left = (a - c)/b
    right = (-a - c)/b

    c_left = left*(crop_sz[0]/2)
    c_right = right*(crop_sz[0]/2)

    left_tmp = np.asarray([-crop_sz[1]/2, c_left])
    right_tmp = np.asarray([crop_sz[1]/2, c_right])
    left, right = extract_hl(left_tmp, right_tmp, org_sz[1])

    return [np.squeeze(left), np.squeeze(right)]

def compute_up_vector(zvp, fovy, eps=1e-7):
    # image size 2 (-1~1)
    focal = 1.0/np.tan(fovy/2.0)
    
    if zvp[2] < 0:
        zvp = -zvp
    zvp = zvp / np.maximum(zvp[2], eps)
    zvp[2] = focal
    return normalize_safe_np(zvp)

def decompose_up_vector(v):
    pitch = np.arcsin(v[2])
    roll = np.arctan(-v[0]/v[1])
    return pitch, roll

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]