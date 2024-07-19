<!---
Copyright 2022 The OFA-Sys Team. 
Copyright 2023 Kai Zhang @ Lehigh. 
Copyright 2024 Shang Luo.
All rights reserved.
This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory.
-->

# MedRIVL

## Installation
```bash
git clone https://github.com/owlsan49/MedRIVL.git
conda env create -f medrivl.yml
python -m pip install pip==21.2.4
pip install -r requirments.txt
```

## Preparing Dataset
We provide the link of raw datasets below.
* RAD: https://vision.aioz.io/f/777a3737ee904924bf0d/?dl=1
* SLAKE: https://www.med-vqa.com/slake/
* PathVQA: https://github.com/UCSD-AI4H/PathVQA/tree/master/data

## Build Retrieved dataset
After downloading the raw datasets, we need to transform them into json format and add q_id for PathVQA. Then build image and text retrieval databases using trainset:
```bash
cd itrs
python rad_test.py
```
Using this program to produce a retrieved trainset, a retrieved validation set and a retrieved testset.

Then we integrate all datasets with images into csv format by:
```bash
cd scripts/preprocess/finetuning
python vqa_rad.py
```
After this process, we get preprocessed dataset on <code>datasets/finetuning/VQA-RAD</code>.

## Finetuning
```bash
cd scripts/vqa
bash train_vqa_rad_beam_scale.sh
```

## Checkpoints
We have prepared checkpoints to facilitate the reproduction of our results.

RAD: [google](https://drive.google.com/file/d/1rfbtWYhMVXUbi-XJxjMRgfAUidsmeN8W/view?usp=sharing);[baidu](https://pan.baidu.com/s/1UcDO6LPuTL0J-FrcF7tc-Q?pwd=362n)

SLAKE: [google](https://drive.google.com/file/d/16KB0H3QJ0AIVKbIBn7RXUK1RqBD3i2VP/view?usp=sharing);[baidu](https://pan.baidu.com/s/1FcAxwjQEQWVWsg7hfrYdHg?pwd=1xsk)

PathVQA: [google](https://drive.google.com/file/d/1gSHwLwgu2-ZE3Few45Hs__QUEUNRUJ6A/view?usp=sharing);[baidu](https://pan.baidu.com/s/1DlQTNd4VtCy13NtPUVlHGw?pwd=jc6x)


## Evaluation
```bash
cd scripts/vqa
bash evaluate_vqa_rad_beam_scale.sh
```

# Note:
We emphasize that MedRIVL are strictly prohibited for Commercial and clinical uses.

# Acknowledgement
* [BiomedGPT](https://github.com/taokz/BiomedGPT?tab=readme-ov-file) Our code is based on BiomedGPT. 
* [PMC-CLIP](https://github.com/WeixiongLin/PMC-CLIP) We use PMC-CLIP as our retrieval embedder.