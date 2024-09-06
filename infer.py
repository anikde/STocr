import json
from PIL import Image
from dataclasses import dataclass
import torch
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
from nltk import edit_distance
from torchvision import transforms as T
from typing import Optional, Callable, Sequence, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

import csv
import os
import sys

def get_transform(img_size: Tuple[int], augment: bool = False, rotation: int = 0):
    transforms = []
    if augment:
        from .augment import rand_augment_transform
        transforms.append(rand_augment_transform())
    if rotation:
        transforms.append(lambda img: img.rotate(rotation, expand=True))
    transforms.extend([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(0.5, 0.5)
    ])
    return T.Compose(transforms)


def load_model(device, checkpoint):
    model = load_from_checkpoint(checkpoint).eval().to(device)
    return model

def get_model_output(device, model, image_path, language):
    hp = model.hparams
    transform = get_transform(hp.img_size, rotation=0)

    image_name = image_path.split("/")[-1]
    img = Image.open(image_path).convert('RGB')
    img = transform(img)
    logits = model(img.unsqueeze(0).to(device))
    probs = logits.softmax(-1)
    preds, probs = model.tokenizer.decode(probs)
    text = model.charset_adapter(preds[0])
    scores = probs[0].detach().cpu().numpy()

    return text

if __name__ =="__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    language = 'hindi'
    checkpoint = f"/DATA/ocr_team_2/anik/model_check_points/hindi.ckpt" 
    test_image_dir = f"/DATA/ocr_team_2/anik/splitonBSTD/bstd/recognition/test/meitei"
    save_path = f"output/{language}_test.json"


    if language != "english":
        model = load_model(device, checkpoint)
    else:
        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)

    parseq_dict = {}
    for image_path in tqdm(image_paths):
        assert os.path.exists(image_path) == True, f"{image_path}"
        text = get_model_output(device, model, image_path, language=f"{language}")
    
        filename = image_path.split('/')[-1]
        parseq_dict[filename] = text

    with open(save_path, 'w') as json_file:
        json.dump(parseq_dict, json_file, indent=4, ensure_ascii=False)