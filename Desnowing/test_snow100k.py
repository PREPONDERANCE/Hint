import os
import math
import argparse
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.HINT_arch import HINT
from skimage.util import img_as_ubyte

parser = argparse.ArgumentParser(description="Image Desnowing using HINT")

parser.add_argument(
    "--input_dir",
    default="./dataset/Snow100K/",
    type=str,
    help="Directory of validation images",
)
parser.add_argument(
    "--result_dir", default="./results/HINT", type=str, help="Directory for results"
)
parser.add_argument(
    "--weights", default="./models/snow100k.pth", type=str, help="Path to weights"
)

args = parser.parse_args()


def splitimage(imgtensor, crop_size=128, overlap_size=64):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs : hs + crop_size, ws : ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (
                    math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-6)
                )
    return score


def mergeimage(split_data, starts, crop_size=128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=True)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs : hs + crop_size, ws : ws + crop_size] += scoremap * simg
        tot_score[:, :, hs : hs + crop_size, ws : ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


####### Load yaml #######
yaml_file = "Options/Desnow_snow100k_HINT.yml"
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode="r"), Loader=Loader)

s = x["network_g"].pop("type")
##########################

model_restoration = HINT(**x["network_g"])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint["params"])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
datasets = ["test2000"]

for dataset in datasets:
    result_dir = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    # inp_dir = os.path.join(args.input_dir, 'test', dataset, 'rain')
    inp_dir = os.path.join(args.input_dir, dataset, "Snow/")
    files = natsorted(
        glob(os.path.join(inp_dir, "*.png")) + glob(os.path.join(inp_dir, "*.jpg"))
    )
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_)) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            B, C, H, W = input_.shape
            corp_size_arg = 256
            overlap_size_arg = 128
            split_data, starts = splitimage(
                input_, crop_size=corp_size_arg, overlap_size=overlap_size_arg
            )
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cpu()
            restored = mergeimage(
                split_data, starts, crop_size=corp_size_arg, resolution=(B, C, H, W)
            )

            restored = (
                torch.clamp(restored, 0, 1)
                .cpu()
                .detach()
                .permute(0, 2, 3, 1)
                .squeeze(0)
                .numpy()
            )

            utils.save_img(
                (
                    os.path.join(
                        result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + ".png"
                    )
                ),
                img_as_ubyte(restored),
            )
