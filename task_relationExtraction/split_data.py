""" 将数据集切割成  train.jsonl val.jsonl """
import os
import re
import json
import random
from task_relationExtraction.conf import args_relationExtraction

args = args_relationExtraction()
random.seed(args.seed)


def convert_data(line):
    head_name, tail_name, relation, text = re.split(r'\t', line)
    match_obj1 = re.search(head_name, text)
    match_obj2 = re.search(tail_name, text)
    if match_obj1 and match_obj2:  # 姑且使用第一个匹配的实体的位置
        head_pos = match_obj1.span()
        tail_pos = match_obj2.span()
        item = {
            'h': {'name': head_name, 'pos': head_pos},
            't': {'name': tail_name, 'pos': tail_pos},
            'relation': relation,
            'text': text
        }
        return item
    else:
        return None


def save_data(lines, file):
    print('保存文件：{}'.format(file))
    unknown_cnt = 0
    with open(file, 'w', encoding='utf-8') as f:
        for line in lines:
            item = convert_data(line)
            if item is None:
                continue
            if item['relation'] == 'unknown':
                unknown_cnt += 1
            json_str = json.dumps(item, ensure_ascii=False)
            f.write('{}\n'.format(json_str))
    print(f'unknown的比例：{unknown_cnt}/{len(lines)}={unknown_cnt/len(lines)}')


def split_data(file):
    file_dir = os.path.dirname(file)
    train_file = os.path.join(file_dir, 'train.jsonl')
    val_file = os.path.join(file_dir, 'val.jsonl')
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    random.shuffle(lines)
    lines_len = len(lines)
    split_len = lines_len * 7 // 10
    train_lines = lines[:split_len]
    val_lines = lines[split_len:]
    save_data(train_lines, train_file)
    save_data(val_lines, val_file)


if __name__ == '__main__':
    all_data = os.path.join(args.data_dir, 'all_data.txt')
    split_data(all_data)
