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
<br></br>

## Preparing Dataset
We provide the link of raw datasets below.
* RAD: https://vision.aioz.io/f/777a3737ee904924bf0d/?dl=1
* SLAKE: https://www.med-vqa.com/slake/
* PathVQA: https://github.com/UCSD-AI4H/PathVQA/tree/master/data

## Build Retrieved dataset
After downloading the raw datasets, we need to transform them into json format and add q_id for PathVQA. Then build image and text retrieval databases using trainset:
```bash
cd scripts/vqa
bash train_vqa_rad_beam_scale.sh
```

## Finetuning
```bash
cd scripts/vqa
bash train_vqa_rad_beam_scale.sh
```

## Checkpoints
We have prepared checkpoints to facilitate the reproduction of our results.

RAD: ...

SLAKE: ...

PathVQA: ...


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