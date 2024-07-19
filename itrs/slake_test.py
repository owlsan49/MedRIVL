# -*- coding: utf-8 -*-
# Copyright (c) 2022, Shang Luo
# All rights reserved.
# 
# Author: 罗尚
# Building Time: 2024/6/7
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
import os

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

model = PMC_CLIP(
    args=args,
    embed_dim=768,
    vision_cfg=vision_cfg,
    text_cfg=text_cfg)


def _convert_to_rgb(image):
    return image.convert('RGB')


ckpt = torch.load(r"C:\Users\LuoShang\Downloads\pmc-clip\checkpoint.pt", map_location='cpu')['state_dict']
# print(ckpt.keys())
# load_info = model.load_state_dict(ckpt, strict=False)
new_ckpt = {}
for kn, vs in ckpt.items():
    if 'module' in kn:
        new_kn = kn[7:]
        new_ckpt[new_kn] = ckpt[kn]
load_info = model.load_state_dict(new_ckpt, strict=True)
print(load_info)
# exit(0)
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

collection0 = client.get_or_create_collection(name="slake_db_img")
collection1 = client.get_or_create_collection(name="slake_db_txt")

image_dir = Path(Path(r'C:\Users\LuoShang\Documents\dataset\Slake1.0\imgs').as_posix())

# with open(r"C:\Users\LuoShang\Documents\dataset\Slake1.0\train.json", encoding='utf-8') as f:
#     annotation = json.load(f)

# exist_annotation = []
#
#
# for ann in annotation:
#     image_path = str(image_dir / ann['img_name'])
#     if os.path.exists(image_path):
#         exist_annotation.append(ann)
# print(f'len ann: {len(exist_annotation)}')

model.to(device)
model.eval()
img_embeddings = []
txt_embeddings = []
batch_size = 128

# imgs = []
# txt_inputs = []
# txt_labels = []
# ann_len = len(exist_annotation)
# for i, ann in enumerate(exist_annotation):
#
#     img = Image.open(str(image_dir / ann['img_name']))
#     imgs.append(transform(img))
#
#     bert_input, bert_label = encode_mlm(
#         caption=ann['question'],
#         vocab=vocab_with_no_special_token,
#         mask_token='[MASK]',
#         pad_token='[PAD]',
#         ratio=0,
#         tokenizer=tokenizer,
#         args=args,
#     )
#     txt_inputs.append(bert_input)
#     txt_labels.append(bert_label)
#
#     if len(imgs) == batch_size or i + 1 == ann_len:
#         imgs_tensor = torch.stack(imgs, dim=0).to(device)
#         batch = {'images': imgs_tensor, 'bert_input': txt_inputs, 'bert_label': txt_labels}
#         with torch.no_grad():
#             embeds = model(batch)
#         img_embeddings.extend(embeds['image_features'].tolist())
#         txt_embeddings.extend(embeds['text_features'].tolist())
#         imgs = []
#         txt_inputs = []
#         txt_labels = []
#
# documents = []
# metadatas = []
# ids = []
# for j, ann in enumerate(exist_annotation):
#     documents.append(ann['answer'])
#     ann.pop("triple")
#     metadatas.append(ann)
#     ids.append(str(ann['qid']))
#
# print(len(documents))
# pace = 30000
# patch = (len(documents) // pace) + 1
# for i in range(patch):
#     collection0.add(
#         documents=documents[i * pace: (i + 1) * pace],
#         embeddings=img_embeddings[i * pace: (i + 1) * pace],
#         metadatas=metadatas[i * pace: (i + 1) * pace],
#         ids=ids[i * pace: (i + 1) * pace])
#     collection1.add(
#         documents=documents[i * pace: (i + 1) * pace],
#         embeddings=txt_embeddings[i * pace: (i + 1) * pace],
#         metadatas=metadatas[i * pace: (i + 1) * pace],
#         ids=ids[i * pace: (i + 1) * pace])

with open(r"C:\Users\LuoShang\Documents\dataset\Slake1.0\validate.json", encoding='utf-8') as f:
    test_anns = json.load(f)
exist_test_anns = []
for ann in test_anns:
    image_path = str(image_dir / ann['img_name'])
    if os.path.exists(image_path) and ann['q_lang'].lower() == 'en':
        exist_test_anns.append(ann)
print(f'len : {len(exist_test_anns)}')
image_threshold = 0.05
test_imgs = []
test_ques_input = []
test_ques_label = []
test_imgs_name = []
q_ids = []
for ann in exist_test_anns:
    image = Image.open(str(image_dir / ann['img_name']))
    image = transform(image)
    q_ids.append(ann['qid'])
    test_imgs_name.append(ann['img_name'])
    test_imgs.append(image)
    ques = ann['question']
    bert_input, bert_label = encode_mlm(
        caption=ques,
        vocab=vocab_with_no_special_token,
        mask_token='[MASK]',
        pad_token='[PAD]',
        ratio=0,
        tokenizer=tokenizer,
        args=args,
    )
    test_ques_input.append(bert_input)
    test_ques_label.append(bert_label)

for i, (img, q_in, q_lab, name, q_id) in enumerate(zip(test_imgs, test_ques_input, test_ques_label, test_imgs_name, q_ids)):
    batch = {'images': img.unsqueeze(0), 'bert_input': [q_in], 'bert_label': [q_lab]}
    with torch.no_grad():
        embeds = model(batch)
        img_embed = embeds['image_features']
        txt_embed = embeds['text_features']
    result0 = collection0.query(
        query_embeddings=img_embed.tolist(),
        n_results=10,
        where={"qid": {"$nin": [q_id]}}
    )
    print(f'======={i}========')
    # print(q_id)
    # print(result0)

    ids = [int(i) for i, dis in zip(result0['ids'][0], result0['distances'][0]) if dis < image_threshold]
    # ids_str = [i for i, dis in zip(result0['ids'][0], result0['distances'][0]) if dis < image_threshold]
    # print(len(ids))
    print(ids)
    if len(ids) > 0:
        try:
            result1 = collection1.query(
                query_embeddings=txt_embed.tolist(),
                n_results=3,
                where={"qid": {"$in": ids}}
            )
            # print(f'---------{i}-------')
            print(result1)
            exist_test_anns[i]['refers'] = result1['documents'][0]
            exist_test_anns[i]['distances'] = result1['distances'][0]
        except RuntimeError:
            exist_test_anns[i]['refers'] = 'none'
            exist_test_anns[i]['distances'] = 'none'
    else:
        exist_test_anns[i]['refers'] = 'none'
        exist_test_anns[i]['distances'] = 'none'


def write_json(file_name, json_data, mode='w'):
    with open(file_name, mode) as jf:
        json.dump(json_data, jf)

write_json(r"C:\Users\LuoShang\Documents\dataset\Slake1.0\validate_refers.json", exist_test_anns)
print(exist_test_anns)

if __name__ == '__main__':
    ...
