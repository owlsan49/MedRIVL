# -*- coding: utf-8 -*-
# Copyright (c) 2022, Shang Luo
# All rights reserved.
# 
# Author: 罗尚
# Building Time: 2024/6/14
# Reference: None
# Description: None
from src.pmc_clip.model.blocks import ModifiedResNet
import torch
import torch.nn as nn
from torchvision.transforms import (Normalize, Compose, Resize, CenterCrop, InterpolationMode,
                                    ToTensor)
from PIL import Image
import chromadb
import json
from pathlib import Path
from dataclasses import dataclass
from src.pmc_clip.model.pmc_clip import PMC_CLIP
from src.pmc_clip.model.config import CLIPVisionCfg, CLIPTextCfg
from src.training.dataset.utils import encode_mlm
from transformers import AutoTokenizer

txt_model_path = r'C:\Users\LuoShang\Downloads\BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

tokenizer = AutoTokenizer.from_pretrained(txt_model_path)
vocab = list(tokenizer.get_vocab().keys())
vocab_with_no_special_token = [vocab_token for vocab_token in vocab if vocab_token not in tokenizer.all_special_tokens]


@dataclass
class MyArgs:
    device = 'cuda'
    mlm = True
    context_length = 77


args = MyArgs()
vision_cfg = CLIPVisionCfg()
vision_cfg.layers = [3, 4, 6, 3]
vision_cfg.width = 64
text_cfg = CLIPTextCfg()
text_cfg.vocab_size = 30522
text_cfg.width = 768
text_cfg.fusion_layers = 4
text_cfg.bert_model_name = txt_model_path

client = chromadb.PersistentClient(path="database")
device = 'cuda'
image_size = 224

# model = PMC_CLIP(
#     args=args,
#     embed_dim=768,
#     vision_cfg=vision_cfg,
#     text_cfg=text_cfg)


def _convert_to_rgb(image):
    return image.convert('RGB')


# ckpt = torch.load(r"C:\Users\LuoShang\Downloads\pmc-clip\checkpoint.pt", map_location='cpu')['state_dict']
# print(ckpt.keys())
# new_ckpt = {}
# for kn, vs in ckpt.items():
#     if 'module' in kn:
#         new_kn = kn[7:]
#         new_ckpt[new_kn] = ckpt[kn]
# load_info = model.load_state_dict(new_ckpt, strict=True)
# print(load_info)

mean = (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
std = (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
normalize = Normalize(mean=mean, std=std)
transform = Compose([
    Resize(image_size, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(image_size),
    _convert_to_rgb,
    ToTensor(),
    normalize,
])

collection0 = client.get_or_create_collection(name="rad_db_img")
collection1 = client.get_or_create_collection(name="rad_db_txt")
c0 = collection0.peek(1)
c1 = collection1.peek(1)
print(c0)
print(c1)
exit(0)
with open(r"C:\Users\LuoShang\Documents\dataset\RAD\trainset.json") as f:
    annotation = json.load(f)
image_dir = Path(Path(r'C:\Users\LuoShang\Documents\dataset\RAD\images').as_posix())
exist_annotation = []
import os

for ann in annotation:
    image_path = str(image_dir / ann['image_name'])
    if os.path.exists(image_path):
        exist_annotation.append(ann)
        # if 'synpic29265.jpg' == ann['image_name']:
        #     print('yesyesyes')

model.to(device)
model.eval()

with open(r"C:\Users\LuoShang\Documents\dataset\RAD\trainset.json") as f:
    test_anns = json.load(f)
exist_test_anns = []
for ann in test_anns:
    image_path = str(image_dir / ann['image_name'])
    if os.path.exists(image_path):
        exist_test_anns.append(ann)

image_threshold = 0.1
image = Image.open(r'C:\Users\LuoShang\Documents\dataset\RAD\images\synpic29265.jpg')
image = transform(image)

ques = "Is there evidence of a pneumothorax?"
bert_input, bert_label = encode_mlm(
        caption=ques,
        vocab=vocab_with_no_special_token,
        mask_token='[MASK]',
        pad_token='[PAD]',
        ratio=0,
        tokenizer=tokenizer,
        args=args,
    )
batch = {'images': image.unsqueeze(0), 'bert_input': [bert_input], 'bert_label': [bert_label]}

with torch.no_grad():
    embeds = model(batch)
    img_embed = embeds['image_features']
    txt_embed = embeds['text_features']

result0 = collection0.query(
    query_embeddings=img_embed.tolist(),
    n_results=10,
    where={"qid": {"$nin": [2]}}
)
print(result0)
# exit(0)
ids = [eval(i) for i, dis in zip(result0['ids'][0], result0['distances'][0]) if dis < image_threshold]

result1 = collection1.query(
            query_embeddings=txt_embed.tolist(),
            n_results=3,
            where={"qid": {"$in": ids}}
        )
print(result1)

if __name__ == '__main__':
    ...
