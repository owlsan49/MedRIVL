# -*- coding: utf-8 -*-
# Copyright (c) 2022, Shang Luo
# All rights reserved.
# 
# Author: 罗尚
# Building Time: 2024/6/29
# Reference: None
# Description: None
import json
import matplotlib.pyplot as plt
mode = 'train'
max_i = 3
def plot_chart(x0, y0, x1, y1, title, mode):
    # 创建折线图
    plt.plot(x0, y0, marker='o', color='blue', label='acc.')
    plt.plot(x1, y1, marker='o', color='red', label='prop.')

    # 添加标题和标签
    plt.title(f"{title} {mode}")
    plt.xlabel('threshold')
    plt.ylabel('percentage')
    plt.legend()
    plt.savefig(f"{title}_{mode}.svg")
    plt.show()



def rad_test():
    # rad rate: 0.4
    with open(f"C:/Users/LuoShang/Documents/dataset/RAD/{mode}set_with_refers.json", encoding='utf-8') as f:
        annotation = json.load(f)

    x0, y0, y1 = [], [], []
    for g in [(u + 1) * 0.05 for u in range(21)]:
        cont = []
        for ann in annotation:
            if isinstance(ann["distances"], list):
                if ann["distances"][0] < g:
                    top_refers = [itm.lower() for itm in ann['refers'][0:max_i]]
                    if str(ann['answer']).lower() in top_refers:
                        cont.append(1)
                    else:
                        cont.append(0)
                    # print(ann['answer'], ann['refers'][0])
        print('=========')
        print(f'g: {g}')
        print(f'len ann: {len(annotation)}')
        print(f'rate: {sum(cont)/len(cont)}')
        print(f'E: {sum(cont) * sum(cont) / len(cont)}')
        print(f'len right: {sum(cont)}')
        print(f'len cont: {len(cont)}')

        x0.append(g)
        y0.append(sum(cont) / len(cont))
        y1.append(len(cont) / len(annotation))

    plot_chart(x0, y0, x0, y1, f'RAD', mode)


def slake_test():
    # slake rate: 0.1
    with open(f"C:/Users/LuoShang/Documents/dataset/Slake1.0/{mode}_refers.json", encoding='utf-8') as f:
        annotation = json.load(f)
    # print(f'len ann: {len(annotation)}')
    exist_annotation = []
    for ann in annotation:
        if ann['q_lang'].lower() == 'en':
            exist_annotation.append(ann)

    x0, y0, y1 = [], [], []
    for g in [(u + 1) * 0.05 for u in range(21)]:
        cont = []
        for ann in exist_annotation:
            if isinstance(ann["distances"], list):
                if ann["distances"][0] < g:
                    top_refers = [itm.lower() for itm in ann['refers'][0:max_i]]
                    if str(ann['answer']).lower() in top_refers:
                        cont.append(1)
                    else:
                        cont.append(0)
                    # print(ann['answer'], ann['refers'][0])
        print('=========')
        print(f'g: {g}')
        print(f'len ann: {len(exist_annotation)}')
        print(f'rate: {sum(cont)/len(cont)}')
        print(f'E: {sum(cont) * sum(cont) / len(cont)}')
        print(f'len right: {sum(cont)}')
        print(f'len cont: {len(cont)}')

        x0.append(g)
        y0.append(sum(cont) / len(cont))
        y1.append(len(cont) / len(exist_annotation))

    plot_chart(x0, y0, x0, y1, f'SLAKE', mode)


def path_test():
    # pathvqa rate:0.14
    with open(f"C:/Users/LuoShang/Documents/dataset/PathVqa/qas/{mode}_ref.json", encoding='utf-8') as f:
        annotation = json.load(f)

    x0, y0, y1 = [], [], []
    for g in [(u+1)*0.05 for u in range(21)]:
        cont = []
        for ann in annotation:
            if isinstance(ann["distances"], list):
                if ann["distances"][0] < g:
                    top_refers = [itm.lower() for itm in ann['refers'][0:max_i]]
                    if str(ann['answer']).lower() in top_refers:
                        cont.append(1)
                    else:
                        cont.append(0)
        if len(cont) == 0:
            cont.append(0)
                # print(ann['answer'], ann['refers'][0])
        print('=========')
        print(f'g: {g}')
        print(f'len ann: {len(annotation)}')
        print(f'rate: {sum(cont) / len(cont)}')
        print(f'E: {sum(cont)*((sum(cont) / len(cont)-0.59))}')
        print(f'len right: {sum(cont)}')
        print(f'len cont: {len(cont)}')

        x0.append(g)
        y0.append(sum(cont) / len(cont))
        y1.append(len(cont) / len(annotation))

    plot_chart(x0, y0, x0, y1, f'PathVQA', mode)

if __name__ == '__main__':
    # rad_test()
    slake_test()
    # path_test()
