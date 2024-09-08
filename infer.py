import csv
import fire
import json
import numpy as np
import os
# import pandas as pd
import sys
import torch


from dataclasses import dataclass
from PIL import Image
from nltk import edit_distance
from torchvision import transforms as T
from typing import Optional, Callable, Sequence, Tuple
from tqdm import tqdm
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint



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

def main(checkpoint, language, image_dir, save_dir):
    """
    Runs the OCR model to process images and save the output as a JSON file.

    Args:
        checkpoint (str): Path to the model checkpoint file.
        language (str): Language code (e.g., 'hindi', 'english').
        image_dir (str): Directory containing the images to process.
        save_dir (str): Directory where the output JSON file will be saved.

    Example usage:
        python your_script.py --checkpoint /path/to/checkpoint.ckpt --language hindi --image_dir /path/to/images --save_dir /path/to/save
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if language != "english":
        model = load_model(device, checkpoint)
    else:
        model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval().to(device)

    parseq_dict = {}
    for image_path in tqdm(os.listdir(image_dir)):
        assert os.path.exists(os.path.join(image_dir, image_path)) == True, f"{image_path}"
        text = get_model_output(device, model, os.path.join(image_dir, image_path), language=f"{language}")
    
        filename = image_path.split('/')[-1]
        parseq_dict[filename] = text

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/{language}_test.json", 'w') as json_file:
        json.dump(parseq_dict, json_file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    fire.Fire(main)