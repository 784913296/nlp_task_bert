# coding=utf-8
from __future__ import absolute_import, division, print_function
from collections import Counter
import os
import copy
import time
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score
import random
import numpy as np
import torch
import re
import jieba

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


# 斯皮尔曼相关性系数/皮尔森相关系数
def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def acc_f1_pea_spea(preds, labels):
    acc_f1 = acc_and_f1(preds, labels)
    pea_spea = pearson_and_spearman(preds,labels)
    return {**acc_f1, **pea_spea}


def flatten(inputs: list) -> list:
    result = []
    [result.extend(line) for line in inputs]
    return result


def get_model_path_list(base_dir):
    """
    从文件夹中获取 pytorch_model.bin 的路径
    """
    model_lists = []
    for root, dirs, files in os.walk(base_dir):
        for _file in files:
            if 'pytorch_model.bin' == _file:
                model_lists.append(os.path.join(root, _file))
    model_lists = sorted(model_lists)
    return model_lists


def swa(model, model_dir, swa_start=1):
    """
    swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
    """
    model_path_list = get_model_path_list(model_dir)
    assert 1 <= swa_start < len(model_path_list) - 1, \
        f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'

    swa_model = copy.deepcopy(model)
    swa_n = 0.

    with torch.no_grad():
        for _ckpt in model_path_list[swa_start:]:
            logger.info(f'Load model from {_ckpt}')
            model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
            tmp_para_dict = dict(model.named_parameters())

            alpha = 1. / (swa_n + 1.)
            for name, para in swa_model.named_parameters():
                para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
            swa_n += 1

    # use 100000 to represent swa to avoid clash
    swa_model_dir = os.path.join(model_dir, f'checkpoint-100000')
    if not os.path.exists(swa_model_dir):
        os.mkdir(swa_model_dir)

    logger.info(f'Save swa model in: {swa_model_dir}')
    swa_model_path = os.path.join(swa_model_dir, 'pytorch_model.bin')
    torch.save(swa_model.state_dict(), swa_model_path)
    return swa_model


def ensemble_vote(labels_list):
    """
    硬投票
    :param labels_list: 所有模型预测出的一个列表
    """
    dict1 = Counter(labels_list)
    max_value = max(dict1.values())

    for k, v in dict1.items():
        if v == max_value:
            max_value_index = int(k)
            break

    return max_value_index



class filterContent:
    """ 处理输入的文本 """

    def __init__(self):
        self.mode_pattern = re.compile(
            r'YJ\w+|1\d{10}|\d{18}|[\U00010000-\U0010ffff]|\w+@\w+.\w+|YT\w+|\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}|\d{4}[-/]\d{2}[-/]\d{2}|\d{4}-\d{1,2}-\d{1,2}|\d{4}.\d{1,2}.\d{1,2}',
            re.A)
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.blankspace_pattern = re.compile(r' {2,}')
        self.cop_pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")

    def mode_replace(self, text):
        return re.sub(self.mode_pattern, ' ', text)

    def url_replace(self, text):
        return re.sub(self.url_pattern, ' ', text)

    def cop_replace(self, text):
        return re.sub(self.cop_pattern, ' ', text)

    def lowercase(self, text):
        return text.lower().strip()

    def blank_space_repalce(self, text):
        return re.sub(self.blankspace_pattern, ' ', text)

    def preprocess(self, text):
        text = self.mode_replace(text)
        text = self.url_replace(text)
        text = self.cop_replace(text)
        text = self.lowercase(text)
        return self.blank_space_repalce(text)

    def word_segment(self, text):
        text = self.preprocess(text)
        # text = ' '.join(jieba.cut(text))
        return self.blank_space_repalce(text)