import os, random, json
import numpy as np
import torch
from task_ner.conf import args_ner


# 将数据集处理成句子列表与对应的标签列表
def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        contents = [_.strip() for _ in f.readlines()]

    # 存放空行所在的行号索引
    index = [-1]
    index.extend([id for id, content in enumerate(contents) if content == ''])
    index.append(len(contents))

    # 按空行分割，读取原文句子及标签
    sentences, labels = [], []
    for j in range(len(index)-1):
        sent, tag = [], []
        start = index[j]+1
        end = index[j+1]
        segment = contents[start:end]
        for line in segment:
            line = line.split()
            if len(line) >1:
                sent.append(line[0])
                tag.append(line[1])
        sentences.append(''.join(sent))
        labels.append(tag)

    # 去除空的句子及标注序列，一般放在末尾
    sentences = [seq for seq in sentences if seq]
    labels = [bio_to_bioes(tag) for tag in labels if tag]
    return sentences, labels


# 将标签转换成 id 并保存
label2id_file = "../task_ner/ner_label2id.json"
train_file = os.path.join(args_ner().data_dir, 'train.txt')
def label2id_save():
    if not os.path.exists(label2id_file):
        print("标签文件不存在，正在根据训练集数据生成标签文件")
        train_sents, train_tags = read_data(train_file)

        # 标签转换成id，并保存成文件
        unique_tags = []
        for tag_list in train_tags:
            for tag in tag_list:
                if tag not in unique_tags:
                    unique_tags.append(tag)

        # 要加上这两个标签
        unique_tags.append("[START]")
        unique_tags.append("[END]")
        label2id = {v: k for k, v in enumerate(unique_tags)}

        with open(label2id_file, "w", encoding="utf-8") as f:
            json.dump(label2id, f)
        return "标签文件生成 ok"


# 读标签文件
def label2id_load():
    label2id_save()
    with open(label2id_file, "r", encoding="utf-8") as f:
        labe2id = json.load(f)
    return labe2id


# BIO 转 BIOES
def bio_to_bioes(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':              # O 直接保留
            new_tags.append(tag)

        elif tag.split('-')[0] == 'B':
            # B不是最后一个,并且紧跟着的后一个是I（B后面是I），直接保留
            if (i + 1) < len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                # B是最后一个或者紧跟着的后一个不是I,那么表示,需要把B换成S表示单字
                new_tags.append(tag.replace('B-', 'S-'))

        elif tag.split('-')[0] == 'I':
            # I 不是最后一个,并且紧跟着的一个是I，直接保留
            if (i + 1) < len(tags) and tags[i+1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                # 是最后一个或者I-ORG后面一个不是以I或B开头的,那么就表示一个词的结尾,就把I换成E表示一个词的结尾
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('非法编码')
    return new_tags


# 从预测的标签列表中获取实体
def get_entity_bioes(sentence, tags):
    item = {"string": sentence, "entities": []}
    entity_name = ""
    idx = 0
    for char, tag in zip(sentence, tags):
        tag_type = tag[2:]
        if tag[0] == "S":
            item["entities"].append({tag_type: char})
            entity_name = ""
        elif tag[0] == "B":
            entity_name += char
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({tag_type: entity_name})
            entity_name = ""
        else:
            entity_name = ""
        idx += 1
    return item['entities']


def get_entity_bio(sentence, tags):
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(sentence):
        if not isinstance(tag, str):
            tag = tags[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(sentence) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(sentence) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, tags, markup='bioes'):
    assert markup in ['bio', 'bioes']
    if markup == 'bio':
        return get_entity_bio(seq, tags)
    else:
        return get_entity_bioes(seq, tags)


# 从预测的标签列表中获取实体，json格式输出
def get_entity_json_bioes(sentence, tags):
    item = {"string": sentence, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(sentence, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item['entities']


# 从预测的标签列表中获取实体，json格式输出
def get_entity_json_bio(sentence, tags):
    item = {"string": sentence, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(sentence, tags):
        if tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item['entities']


def get_entitys_json(seq, tags, markup='bioes'):
    assert markup in ['bio', 'bioes']
    if markup == 'bio':
        return get_entity_json_bio(seq, tags)
    else:
        return get_entity_json_bioes(seq, tags)


