#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082

user_dir=../../module
bpe_dir=../../utils/BPE

# val or test
split=$1

data_dir=../../datasets/finetuning/Slake
data=${data_dir}/test.tsv
ans2label_file=${data_dir}/trainval_ans2label.pkl

declare -a Scale=('base')  # 'tiny' 'medium' 'base'

for scale in ${Scale[@]}; do
    if [[ $scale =~ "tiny" ]]; then
        patch_image_size=256
    elif [[ $scale =~ "medium" ]]; then
        patch_image_size=256
    elif [[ $scale =~ "base" ]]; then  
        patch_image_size=384
    fi

#    path=/root/autodl-tmp/biomedgpt/slake.pt
    path=/root/autodl-tmp/project/checkpoints/tuned_checkpoints/Slake/base/100_0.04_1e-4_384_/checkpoint-rag2.pt
    result_path=./results/vqa_slake_beam/${scale}
    mkdir -p $result_path
    selected_cols=0,7,2,3,4,5,6

    log_file=${result_path}/${scale}".log"
    # log_file=${result_path}/"val_"${scale}".log"

    CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=${MASTER_PORT} ../../evaluate.py \
        ${data} \
        --path=${path} \
        --user-dir=${user_dir} \
        --task=vqa_gen \
        --batch-size=64 \
        --log-format=simple --log-interval=100 \
        --seed=7 \
        --gen-subset=${split} \
        --results-path=${result_path} \
        --fp16 \
        --ema-eval \
        --beam-search-vqa-eval \
        --beam=1 \
        --unnormalized \
        --temperature=1.0 \
        --num-workers=0 \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}" > ${log_file} 2>&1
done