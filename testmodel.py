import os

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor

from model.dataset_forTesting import BokehDataset
from model.metrics import calculate_lpips, calculate_psnr, calculate_ssim
from model.bokeh import bokeh
import datetime
from PIL import Image

to_tensor = ToTensor()
to_pil = ToPILImage()
lpips_list = []
ssim_list = []
psnr_list = []

PATH = " type model path here "
val_path = 'type validation set path here'

device = torch.device("cuda:0")
bokehnet = torch.load(PATH,map_location=device).module


from os import walk
pic_list = []

for f, _, i in walk(val_path):
    for j in i:
        if 'txt' in j or 'png' in j or 'tgt' in j:
            continue
        else:
            pic_list.append(j)
meta = {}
with open(val_path+"/meta.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
    meta[id] = (src_lens, tgt_lens, disparity)

trans = ToTensor()

for picture in pic_list:
    id = picture.split(".")[0]
    pic4inference = Image.open(val_path+"/"+str(picture))
    target  = Image.open(val_path+"/"+str(id)+'.tgt.jpg')
    pic4inference = trans(pic4inference)
    target = trans(target)
    
    src_lens, tgt_lens, disparity = meta[id]

    if src_lens == "Sony50mmf1.8BS":
        src_lens = [1.8]
    elif src_lens == "Sony50mmf16.0BS":
        src_lens = [16.0]
    elif src_lens == "Canon50mmf1.8BS":
        src_lens = [-1.8]
    elif src_lens == "Canon50mmf1.4BS":
        src_lens = [-1.4]
    elif src_lens == "Sony50mmf1.4BS":
        src_lens = [1.4]
    src_lens = torch.FloatTensor(src_lens).to(device)
    
    if tgt_lens == "Sony50mmf1.8BS":
        tgt_lens = [1.8]
    elif tgt_lens == "Sony50mmf16.0BS":
        tgt_lens = [16.0]
    elif tgt_lens == "Canon50mmf1.8BS":
        tgt_lens = [-1.8]
    elif tgt_lens == "Canon50mmf1.4BS":
        tgt_lens = [-1.4]
    elif tgt_lens == "Sony50mmf1.4BS":
        tgt_lens = [1.4]
    tgt_lens = torch.FloatTensor(tgt_lens).to(device)

    disparity = [int(disparity)]
    disparity = torch.FloatTensor(disparity).to(device)

    #print(pic4inference.shape)
    pic4inference = pic4inference.to(device)
    target = target.to(device)

    #pic4inference = img_transform(pic4inference).unsqueeze(0)
    pic4inference = pic4inference.unsqueeze(0)
    with torch.no_grad():
        output = bokehnet(pic4inference,src_lens,tgt_lens,disparity)
    target = target.unsqueeze(0)

    # Calculate metrics
    lpips = np.mean([calculate_lpips(img0, img1) for img0, img1 in zip(output, target)])
    psnr = np.mean(
        [calculate_psnr(np.asarray(to_pil(img0)), np.asarray(to_pil(img1))) for img0, img1 in zip(output, target)]
    )
    ssim = np.mean(
        [calculate_ssim(np.asarray(to_pil(img0)), np.asarray(to_pil(img1))) for img0, img1 in zip(output, target)]
    )
    lpips_list.append(lpips)
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    print(id,f"Metrics: lpips={lpips:0.03f}, psnr={psnr:0.03f}, ssim={ssim:0.03f}")

lpips = sum(lpips_list)/len(lpips_list)
psnr = sum(psnr_list)/len(psnr_list)
ssim = sum(ssim_list)/len(ssim_list)
print("Result:",f"Metrics: lpips={lpips:0.03f}, psnr={psnr:0.03f}, ssim={ssim:0.03f}")

