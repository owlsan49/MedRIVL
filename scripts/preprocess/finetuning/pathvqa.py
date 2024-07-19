import os
from PIL import Image
from io import BytesIO
import base64
import json
import pickle

qa_dir = '../../../datasets/PathVQA/qas'
img_dir = '../../../datasets/PathVQA/images'
output_dir = '../../../datasets/finetuning/PathVQA'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def extraction(mode):
    if mode == 'train':
        path = os.path.join(qa_dir, 'train_ref.json')
    elif mode == 'val':
        path = os.path.join(qa_dir, 'val_ref.json')
    elif mode == 'test':
        path = os.path.join(qa_dir, 'test_ref.json')

    output_file_name = os.path.join(output_dir, mode + '.tsv')
    index = 0

    with open(output_file_name, 'w') as out:
        with open(path, "rb") as input_file:
            data = json.load(input_file)

        for item in data:
            img_id = item['image']
            question_id = str(item['qid'])
            question = item['question']
            confident_ans = '1|!+' + item['answer']

            if 'refers' in item.keys() and isinstance(item['refers'], list):
                refers = [ref.replace('\t', '').lower().strip() for ref in item['refers']]
                refers = '|'.join(refers)
                dis = [str(di) for di in item['distances']]
                distances = '|'.join(dis)
            else:
                refers = 'none'
                distances = 'none'

            # image string64base
            img_path = os.path.join(img_dir, img_id + '.jpg')
            img = Image.open(img_path)
            img_buffer = BytesIO()
            img.save(img_buffer, format=img.format)
            byte_data = img_buffer.getvalue()
            base64_str = base64.b64encode(byte_data)
            base64_str = base64_str.decode("utf-8")

            # question_id, image_id, question, answer (with confidence), predicted object labels (set to empty string), image (base64 string)
            out.write(question_id + '\t' + img_id + '\t' + question + '\t' + confident_ans + '\t' + str(
                '') + '\t' + refers + '\t' + distances + '\t' + base64_str + '\n')
            index += 1
            if index % 1000 == 0:
                print("finish '{}' instance {}".format(mode, index))

    print("Completed! totally {} '{}' instances".format(index, mode))
    return index


def ans2label():
    path_train = os.path.join(qa_dir, 'train_ref.json')
    path_val = os.path.join(qa_dir, 'val_ref.json')

    output_file_name = os.path.join(output_dir, 'trainval_ans2label.pkl')
    index = 0

    with open(output_file_name, 'w') as out:
        ans2label = {}

        with open(path_train, "rb") as input_file:
            data = json.load(input_file)

        for item in data:
            confident_ans = item['answer']
            if confident_ans not in ans2label.keys():
                ans2label[confident_ans] = index
                index += 1
                if index % 100 == 0:
                    print("finish labeling {} answers".format(index))

        with open(path_val, "rb") as input_file:
            data = json.load(input_file)

        for item in data:
            confident_ans = item['answer']
            if confident_ans not in ans2label.keys():
                ans2label[confident_ans] = index
                index += 1
                if index % 100 == 0:
                    print("finish labeling {} answers".format(index))

        with open(output_file_name, 'wb') as f:
            pickle.dump(ans2label, f)

    return index


if __name__ == '__main__':
    total = 0
    for mode in ['train', 'val', 'test']:  # 'train', 'val', 'test'
        num = extraction(mode)
        total += num
    print("Completed! totally {} instances".format(total))

# num = ans2label()
# print("Completed! totally {} answers".format(num))



