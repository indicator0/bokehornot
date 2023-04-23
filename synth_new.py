from __future__ import absolute_import, division, print_function
import numpy as np
import PIL.Image as Image
import torch
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage, ToTensor, PILToTensor
from model.bokeh import bokeh
from PIL import Image
from tqdm import tqdm
import PIL.ImageChops as chops
import time

device = torch.device("cuda")

bokehnet = bokeh().to(device)
to_pil = ToPILImage()

PATH = "type model path here"
val_path = 'folder path with images you want to inference'

bokehnet = torch.load(PATH,map_location=device)
bokehnet.eval()
print("weights loaded!!")

from os import walk
pic_list = []

for f, _, i in walk(val_path):
    for j in i:
        if 'txt' in j or 'tgt' in j or 'alpha' in j:
            continue
        else:
            pic_list.append(j)
meta = {}
with open(val_path+"/meta.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
    meta[id] = (src_lens, tgt_lens, disparity)

trans = transforms.Compose([transforms.ToTensor()])
transform = transforms.Compose([
    transforms.PILToTensor()])

timelapse1 = []
timelapse2 = []

for picture in pic_list:
    T1 = time.perf_counter()
    pic4inference = Image.open(val_path+"/"+str(picture))
    pic4inference_src  = Image.open(val_path+"/"+str(picture))
    pic4inference = trans(pic4inference)
    id = picture.split(".")[0]
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
        src_lens = [1.8]
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
        tgt_lens = [1.8]
    tgt_lens = torch.FloatTensor(tgt_lens).to(device)

    disparity = [int(disparity)]
    disparity = torch.FloatTensor(disparity).to(device)

    pic4inference = pic4inference.to(device)

    with torch.no_grad():
        T2 = time.perf_counter()
        bok_pred = bokehnet(pic4inference,src_lens,tgt_lens,disparity)
        T3 = time.perf_counter()

    bok_pred = bok_pred.detach().cpu().squeeze(0)

    result = bok_pred
    T4 = time.perf_counter()

    result = to_pil(result.clip(0,1))

    result.save('type output path here'+str(picture.split('.')[0])+'.src.png')
    timelapse1.append(T4-T1)
    timelapse2.append(T3-T2)

print("Avg Synth Time:",sum(timelapse1)/len(timelapse1))
print("Avg Inference Time:",sum(timelapse2)/len(timelapse2))
