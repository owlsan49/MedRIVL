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
import copy
import random
from tqdm import tqdm

txt_model_path = r'C:\Users\LuoShang\Downloads\BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
train_data_path = Path(Path(r"C:\Users\LuoShang\Documents\dataset\PathVqa\qas\train_qid.json").as_posix())
val_data_path = Path(Path(r"C:\Users\LuoShang\Documents\dataset\PathVqa\qas\val_qid.json").as_posix())
test_data_path = Path(Path(r"C:\Users\LuoShang\Documents\dataset\PathVqa\qas\test_qid.json").as_posix())
image_dir = Path(Path(r'C:\Users\LuoShang\Documents\dataset\PathVqa\images').as_posix())


save_train_path = Path(Path(r"C:\Users\LuoShang\Documents\dataset\PathVqa\qas\train_ref.json").as_posix())
save_val_path = Path(Path(r"C:\Users\LuoShang\Documents\dataset\PathVqa\qas\val_ref.json").as_posix())
save_test_path = Path(Path(r"C:\Users\LuoShang\Documents\dataset\PathVqa\qas\test_ref.json").as_posix())


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
new_ckpt = {}
for kn, vs in ckpt.items():
    if 'module' in kn:
        new_kn = kn[7:]
        new_ckpt[new_kn] = ckpt[kn]
load_info = model.load_state_dict(new_ckpt, strict=True)
print(load_info)

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

# client.delete_collection(name="path_db_img")
# client.delete_collection(name="path_db_txt")
# exit(0)
collection0 = client.get_or_create_collection(name="path_db_img")
collection1 = client.get_or_create_collection(name="path_db_txt")

# with open(str(train_data_path), encoding='utf-8') as f:
#     annotation = json.load(f)
#
# exist_annotation = []
#
#
# for ann in annotation:
#     image_path = str(image_dir / (ann['image'] + '.jpg'))
#     if os.path.exists(image_path):
#         exist_annotation.append(ann)
# print(f'len ann: {len(exist_annotation)}')

model.to(device)
model.eval()
batch_size = 128

# img_embeddings = []
# txt_embeddings = []
# ext_metadatas = []
# ext_documents = []
# ext_ids = []
# ext_img_embeddings = []
# ext_txt_embeddings = []
#
# tmp_metadatas = []
# tmp_documents = []
# tmp_ids = []
#
# imgs = []
# txt_inputs = []
# txt_labels = []
# ann_len = len(exist_annotation)
# for i, ann in enumerate(exist_annotation):
#
#     img = Image.open(str(image_dir / (ann['image'] + '.jpg')))
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
#     tmp_documents.append(ann['answer'])
#     tmp_ann = copy.deepcopy(ann)
#     tmp_ann['qid'] = 'ext_'+str(ann['qid'])
#     tmp_metadatas.append(tmp_ann)
#     tmp_ids.append(tmp_ann['qid'])
#
#     if len(imgs) == batch_size or i + 1 == ann_len:
#         imgs_tensor = torch.stack(imgs, dim=0).to(device)
#         batch = {'images': imgs_tensor, 'bert_input': txt_inputs, 'bert_label': txt_labels}
#         with torch.no_grad():
#             embeds = model(batch)
#         img_embeddings.extend(embeds['image_features'].tolist())
#         txt_embeddings.extend(embeds['text_features'].tolist())
#         if random.random() < 0.8:
#             ext_metadatas.extend(tmp_metadatas)
#             ext_documents.extend(tmp_documents)
#             ext_ids.extend(tmp_ids)
#             ext_img_embeddings.extend(embeds['image_features'].tolist())
#             ext_txt_embeddings.extend((embeds['text_features']+torch.randn(len(embeds['text_features']), 768).cuda()*0.001).tolist())
#
#         imgs = []
#         txt_inputs = []
#         txt_labels = []
#
#         tmp_metadatas = []
#         tmp_documents = []
#         tmp_ids = []
#
#
#
# documents = []
# metadatas = []
# ids = []
# for j, ann in enumerate(exist_annotation):
#     documents.append(ann['answer'])
#     ann['qid'] = str(ann['qid'])
#     metadatas.append(ann)
#     ids.append(ann['qid'])
#
# documents.extend(ext_documents)
# img_embeddings.extend(ext_img_embeddings)
# metadatas.extend(ext_metadatas)
# ids.extend(ext_ids)
# txt_embeddings.extend(ext_txt_embeddings)
#
# print(len(documents))

# print(documents[99])
# print(documents[-1])
# print(metadatas[99])
# print(metadatas[-1])
# print(ids[99])
# print(ids[-1])
# print(img_embeddings[99])
# print(img_embeddings[-1])
# print(txt_embeddings[99])
# print(txt_embeddings[-1])
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



# write referance on data_json
with open(str(train_data_path), encoding='utf-8') as f:
    data_anns = json.load(f)
exist_data_anns = []
for ann in data_anns:
    image_path = str(image_dir / (ann['image'] + '.jpg'))
    if os.path.exists(image_path):
        exist_data_anns.append(ann)

print(f'2. len ann: {len(exist_data_anns)}')

image_threshold = 0.1
data_imgs = []
data_ques_input = []
data_ques_label = []
data_imgs_name = []
q_ids = []
img_embeddings = []
txt_embeddings = []
for i, ann in tqdm(enumerate(exist_data_anns), total=len(exist_data_anns)):
    image = Image.open(str(image_dir / (ann['image'] + '.jpg')))
    image = transform(image)
    q_ids.append(ann['qid'])
    data_imgs_name.append(ann['image'])
    data_imgs.append(image)
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
    data_ques_input.append(bert_input)
    data_ques_label.append(bert_label)
    if len(data_imgs) == batch_size or i + 1 == len(exist_data_anns):
        imgs_tensor = torch.stack(data_imgs, dim=0).to(device)
        batch = {'images': imgs_tensor, 'bert_input': data_ques_input, 'bert_label': data_ques_label}
        with torch.no_grad():
            embeds = model(batch)
        img_embeddings.extend(embeds['image_features'].tolist())
        txt_embeddings.extend(embeds['text_features'].tolist())
        data_imgs = []
        data_ques_input = []
        data_ques_label = []
print(f'img_embeddings: {len(img_embeddings)}')
print(f'txt_embeddings: {len(txt_embeddings)}')
print(f'data_imgs_name: {len(data_imgs_name)}')
print(f'q_ids: {len(q_ids)}')
tqdm_bar = tqdm(enumerate(zip(img_embeddings, txt_embeddings, data_imgs_name, q_ids)), total=len(q_ids))
for i, (img_emb, txt_emb, name, q_id) in tqdm_bar:
    if random.random() < 0.4:
        result0 = collection0.query(
            query_embeddings=img_emb,
            n_results=20,
            where={"qid": {"$nin": [str(q_id)]}}
        )
        # print(q_id)
        # print(result0)

        ids = [i for i, dis in zip(result0['ids'][0], result0['distances'][0]) if dis < image_threshold]
        # print(ids)
        if len(ids) > 0:
            try:
                result1 = collection1.query(
                    query_embeddings=txt_emb,
                    n_results=1,
                    where={"qid": {"$in": ids}}
                )
                # print(f'---------{i}-------')
                tqdm_bar.set_postfix({"ori_id": q_id, "search_id": result1['ids'][0][0]})
                exist_data_anns[i]['refers'] = result1['documents'][0]
                exist_data_anns[i]['distances'] = result1['distances'][0]
            except RuntimeError:
                exist_data_anns[i]['refers'] = 'none'
                exist_data_anns[i]['distances'] = 'none'
        else:
            exist_data_anns[i]['refers'] = 'none'
            exist_data_anns[i]['distances'] = 'none'
    else:
        exist_data_anns[i]['refers'] = 'none'
        exist_data_anns[i]['distances'] = 'none'


def write_json(file_name, json_data, mode='w'):
    with open(file_name, mode) as jf:
        json.dump(json_data, jf)

write_json(str(save_train_path), exist_data_anns)

if __name__ == '__main__':
    ...
