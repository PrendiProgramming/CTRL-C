import os
import os.path as osp
import argparse
from datetime import date
import json
import random
import time
from pathlib import Path
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
from datasets import build_holicity_dataset
from models import build_model
from config import cfg

cmap = plt.get_cmap("jet")
norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

def c(x):
    return sm.to_rgba(x)

def get_args_parser():
    parser = argparse.ArgumentParser('Set gptran', add_help=False)
    parser.add_argument('--config-file', 
                        metavar="FILE",
                        help="path to config file",
                        type=str,
                        default='config-files/gptran.yaml')
    parser.add_argument("--opts",
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER
                        )
    return parser

def compute_vp_err(vp1, vp2, dim=-1):
    cos_sim = F.cosine_similarity(vp1, vp2, dim=dim).abs()
    cos_sim = np.clip(cos_sim.item(), 0.0, 1.0)    
    return np.degrees(np.arccos(cos_sim))

def compute_hl_np(hl, sz, eps=1e-6):
    (a,b,c) = hl
    if b < 0:
        a, b, c = -a, -b, -c
    b = np.maximum(b, eps)
    
    left = np.array([-1.0, (a - c)/b])        
    right = np.array([1.0, (-a - c)/b])

    # scale back to original image    
    scale = sz[1]/2
    left = scale*left
    right = scale*right
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

def cosine_similarity(v1, v2, eps=1e-7):
    v1 = v1 / np.maximum(LA.norm(v1), eps)
    v2 = v2 / np.maximum(LA.norm(v2), eps)
    return np.sum(v1*v2)

def normalize_safe_np(v, eps=1e-7):
    return v/np.maximum(LA.norm(v), eps)

def compute_up_vector_error(pred_zvp, pred_fovy, target_up_vector):
    pred_up_vector = compute_up_vector(pred_zvp, pred_fovy)
    cos_sim = cosine_similarity(target_up_vector, pred_up_vector)

    target_pitch, target_roll = decompose_up_vector(target_up_vector)

    if cos_sim < 0:
        pred_pitch, pred_roll = decompose_up_vector(-pred_up_vector)
    else:
        pred_pitch, pred_roll = decompose_up_vector(pred_up_vector)

    err_angle = np.degrees(np.arccos(np.abs(cos_sim)))
    err_pitch = np.degrees(np.abs(pred_pitch - target_pitch))
    err_roll = np.degrees(np.abs(pred_roll - target_roll))
    return err_angle, err_pitch, err_roll

def compute_fovy_error(pred_fovy, target_fovy):
    pred_fovy = np.degrees(pred_fovy)
    target_fovy = np.degrees(target_fovy)
    err_fovy = np.abs(pred_fovy - target_fovy)
    return err_fovy.item()

def compute_horizon_error(pred_hl, target_hl, img_sz):
    target_hl_pts = compute_hl_np(target_hl, img_sz)
    pred_hl_pts = compute_hl_np(pred_hl, img_sz)
    err_hl = np.maximum(np.abs(target_hl_pts[0][1] - pred_hl_pts[0][1]),
                        np.abs(target_hl_pts[1][1] - pred_hl_pts[1][1]))
    err_hl /= img_sz[0] # height
    return err_hl
    
def draw_attention(img, weights, cmap, savepath):
    extent = [-1, 1, 1, -1]
    num_layer = len(weights)
    plt.figure(figsize=(num_layer*3,3))
    for idx_l in range(num_layer):                    
        plt.subplot(1, num_layer, idx_l + 1)
        plt.imshow(img, extent=extent)
        plt.imshow(weights[idx_l], cmap=cmap, alpha=0.3, extent=extent)
        plt.axis('off')
    plt.savefig(savepath, pad_inches=0, bbox_inches='tight')
    plt.close('all')

def draw_attention_segs(img, weights, segs, cmap, savepath):
    num_layer = len(weights)
    num_segs = len(segs)
    plt.figure(figsize=(num_layer*3,3))
    for idx_l in range(num_layer):                    
        plt.subplot(1, num_layer, idx_l + 1)
        plt.imshow(img, extent=[-1, 1, 1, -1])                 
        ws = weights[idx_l]
        ws = (ws - ws.min())/(ws.max() - ws.min())
        for idx_s in range(num_segs):
            plt.plot((segs[idx_s,0], segs[idx_s,2]), 
                     (segs[idx_s,1], segs[idx_s,3]), c=cmap(ws[idx_s]))
        plt.axis('off')
    plt.savefig(savepath, pad_inches=0, bbox_inches='tight')
    plt.close('all')    
    
def to_device(data, device):
    if type(data) == dict:
        return {k: v.to(device) for k, v in data.items()}
    return [{k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()} for t in data]    

def main(cfg):
    device = torch.device(cfg.DEVICE)
    
    model, _ = build_model(cfg)
    model.to(device)
    
    dataset_test = build_holicity_dataset(image_set='test', cfg=cfg)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=2)
    
    output_dir = Path(cfg.OUTPUT_DIR)
    
    checkpoint = torch.load('logs/checkpoint.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    
    # initlaize for visualization
    name = f'holicity_test_{date.today()}'
    if cfg.TEST.DISPLAY:
        fig_output_dir = osp.join(output_dir,'{}'.format(name))
        os.makedirs(fig_output_dir, exist_ok=True)
        
    for i, (samples, extra_samples, targets) in enumerate(tqdm(data_loader_test)):
        with torch.no_grad():
            samples = samples.to(device)
            extra_samples = to_device(extra_samples, device)
            outputs, extra_info = model(samples, extra_samples)
        
            pred_zvp = outputs['pred_zvp'].to('cpu')[0].numpy()
            pred_fovy = outputs['pred_fovy'].to('cpu')[0].numpy()
            pred_hl = outputs['pred_hl'].to('cpu')[0].numpy()
            
            pred_vweight = outputs['pred_vline_logits'].sigmoid()
            pred_vweight = pred_vweight.to('cpu')[0].numpy()
            pred_vweight = np.squeeze(pred_vweight, axis=1)
            
            pred_hweight = outputs['pred_hline_logits'].sigmoid()
            pred_hweight = pred_hweight.to('cpu')[0].numpy()
            pred_hweight = np.squeeze(pred_hweight, axis=1)
            
            aux_outputs = outputs['aux_outputs']
                        
            img_sz = targets[0]['org_sz']
            filename = targets[0]['filename']
            filename = osp.splitext(filename)[0]
                
            target_up_vector = targets[0]['up_vector'].numpy()
            target_zvp = targets[0]['zvp'].numpy()
            target_fovy = targets[0]['fovy'].numpy()
            target_hl = targets[0]['hl'].numpy()
            
            if pred_zvp[2] < 0:
                pred_zvp = -pred_zvp
            target_zvp = normalize_safe_np(target_zvp)
            
            target_segs = targets[0]['segs'].numpy()
            target_lines = targets[0]['lines'].numpy()
            target_mask = targets[0]['line_mask'].numpy()
            
            if cfg.TEST.DISPLAY:
                os.makedirs(osp.dirname(osp.join(fig_output_dir, filename)), 
                                exist_ok=True)
                img = targets[0]['org_img']
                lowtone_img = np.array(img)//4 + 190
                h,w = lowtone_img.shape[:2]

                num_layers = cfg.MODELS.TRANSFORMER.ENC_LAYERS
                num_aux_outputs = num_layers - 1
                
                plt.figure(figsize=(5,5))
                plt.imshow(img, extent=[-1, 1, 1, -1])                                 
                plt.xlim(-1, 1)
                plt.ylim( 1,-1)
                plt.axis('off')
                plt.savefig(osp.join(fig_output_dir, filename+'.jpg'),  
                            pad_inches=0, bbox_inches='tight')
                plt.close('all')
                
                # draw zvp
                plt.figure(figsize=(5,5))
                plt.imshow(lowtone_img, extent=[-1, 1, 1, -1])                 
                plt.plot((0, target_zvp[0]), (0, target_zvp[1]), 'r-', alpha=1.0)
                plt.plot((0, pred_zvp[0]), (0, pred_zvp[1]), 'g-', alpha=1.0)  
                plt.xlim(-1, 1)
                plt.ylim( 1,-1)
                plt.axis('off')
                plt.savefig(osp.join(fig_output_dir, filename+'_zvp.jpg'),  
                            pad_inches=0, bbox_inches='tight')
                plt.close('all')

            # draw horizon line
                img_sz = targets[0]['org_sz']
                target_hl_pts = compute_hl_np(target_hl, img_sz)
                pred_hl_pts = compute_hl_np(pred_hl, img_sz)

                plt.figure(figsize=(5,5))
            #                 plt.title(f'horzon error {err_hl}')
                plt.imshow(lowtone_img, 
                        extent=[-img_sz[1]/2, img_sz[1]/2, img_sz[0]/2, -img_sz[0]/2])
                plt.plot([target_hl_pts[0][0], target_hl_pts[1][0]], 
                        [target_hl_pts[0][1], target_hl_pts[1][1]], 'r-', alpha=1.0)
            plt.plot([pred_hl_pts[0][0], pred_hl_pts[1][0]], 
                    [pred_hl_pts[0][1], pred_hl_pts[1][1]], 'g-', alpha=1.0)
            plt.plot((0, pred_zvp[0]*img_sz[1]), (0, pred_zvp[1]*img_sz[0]), 'r-', alpha=1.0)  
            plt.xlim(-img_sz[1]/2, img_sz[1]/2)
            plt.ylim( img_sz[0]/2, -img_sz[0]/2)
            plt.axis('off')
            plt.savefig(osp.join(fig_output_dir, filename+'_hl.jpg'),  
                        pad_inches=0, bbox_inches='tight')
            plt.close('all')

            # visualize line weights
            num_segs = int(target_mask.sum())
            segs = target_segs[:num_segs]
                            
            plt.figure(figsize=(5,5))                
            plt.imshow(np.array(img)//2 + 127, extent=[-1, 1, 1, -1])                
            plt.plot((segs[:,0], segs[:,2]), (segs[:,1], segs[:,3]), 
                        '-r', alpha=0.5)
            plt.xlim(-1, 1)
            plt.ylim( 1,-1)
            plt.axis('off')
            plt.savefig(osp.join(fig_output_dir, filename+'_segs.jpg'),  
                        pad_inches=0, bbox_inches='tight')
            plt.close('all')

            vw = pred_vweight
            hw = pred_hweight

            plt.figure(figsize=(5,5))                
            #                 plt.title('zenith vp lines')
            plt.imshow(lowtone_img, extent=[-1, 1, 1, -1])
            for i in range(num_segs):
                plt.plot((segs[i,0], segs[i,2]), (segs[i,1], segs[i,3]), 
                            c=c(vw[i]), alpha=1.0)
            plt.xlim(-1, 1)
            plt.ylim( 1,-1)
            plt.axis('off')
            plt.savefig(osp.join(fig_output_dir, filename+'_lines_v.jpg'),  
                        pad_inches=0, bbox_inches='tight')
            plt.close('all')

            plt.figure(figsize=(5,5))                
            #                 plt.title('horizon vps lines')
            plt.imshow(lowtone_img, extent=[-1, 1, 1, -1])
            for i in range(num_segs):
                plt.plot((segs[i,0], segs[i,2]), (segs[i,1], segs[i,3]), 
                            c=c(hw[i]), alpha=1.0)
            plt.xlim(-1, 1)
            plt.ylim( 1,-1)
            plt.axis('off')
            plt.savefig(osp.join(fig_output_dir, filename+'_lines_h.jpg'),  
                        pad_inches=0, bbox_inches='tight')
            plt.close('all')
            
#         if i > 100:
#             break

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPANet training and evaluation script', 
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    main(cfg)