#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import cv2
import numpy as np
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import mediapy as media
from scene.cameras import Camera

def save_torch_image(path, image_tensor):
    image = torch.clip(image_tensor, 0., 1.).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration))

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        save_torch_image(os.path.join(render_path, f'{idx:05d}' + ".png"), rendering)
    
    first_frame = cv2.imread(os.path.join(render_path, '00000.png'))
    video_path = os.path.join(model_path, name, 'video.mp4')
    with media.VideoWriter(video_path, shape=first_frame.shape[:2], fps=30, crf=18) as writer:
        for frame_id in range(len(views)):
            frame = cv2.imread(os.path.join(render_path, f'{frame_id:05d}.png'))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.add_image(frame)
    
    # 删除图片，这个文件夹太大了
    os.system(f'rm -r {render_path}')
    
def get_video_cameras(poses: np.ndarray, train_cameras):
    video_cameras = []
    ref_camera = train_cameras[0]
    FoVx = ref_camera.FoVx
    FoVy = ref_camera.FoVy
    image = torch.ones_like(ref_camera.original_image)
    data_device = ref_camera.data_device
    for i, c2w in enumerate(poses):
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        cam = Camera(colmap_id=i, R=R, T=T,
                     FoVx=FoVx, FoVy=FoVy, 
                     image=image, gt_alpha_mask=None,
                     image_name=f'video_{i}', uid=i,
                     data_device=data_device)
        
        video_cameras.append(cam)
        
    return video_cameras

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, render_poses_path: str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, 
                   "video", 
                   scene.loaded_iter, 
                   get_video_cameras(np.load(render_poses_path), scene.getTrainCameras()), 
                   gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_poses_path", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.render_poses_path)