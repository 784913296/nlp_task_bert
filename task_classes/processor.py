# coding:utf-8
import os, logging
import torch
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from collections import defaultdict
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from tqdm import tqdm


logger = logging.getLogger(__name__)


def load_dict(id_file):
    """ 将  cid2index.txt index2cid.txt 装载成字典格式 : {int: int} """
    _dict = defaultdict()
    lines = open(id_file, encoding="utf-8").readlines()
    for line in lines:
        key, value = line.strip().split("\t")
        _dict.update({int(key): int(value)})
    return _dict

def load_cid2name(cid2name_file):
    """ 将  cid2name.txt 装载成字典格式: {int: srt} """
    _dict = defaultdict()
    lines = open(cid2name_file, encoding="utf-8").readlines()
    for line in lines:
        key, value = line.strip().split("\t")
        _dict.update({int(key): value})
    return _dict


class classesProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train.txt'))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test.txt'))

    def get_index2cid(self):
        index2cid = load_dict("../data/task_classes/index2cid.txt")
        return index2cid

    def get_cid2index(self):
        cid2index = load_dict("../data/task_classes/cid2index.txt")
        return cid2index

    def get_cid2name(self):
        cid2name = load_cid2name("../data/task_classes/cid2name.txt")
        return cid2name

    def _create_examples(self, data_file):
        examples = []
        with open(data_file, mode='r', encoding='utf8') as f:
            for id, line in tqdm(enumerate(f.readlines())):
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue

                text, cid = line[0], int(line[1])
                text = ILLEGAL_CHARACTERS_RE.sub(r'', text)

                # 将 cid 转换成 index
                cid2index = self.get_cid2index()
                if cid in cid2index:
                    label = cid2index[cid]
                    example = InputExample(guid=id, text_a=text, label=label)
                    examples.append(example)
        return examples


def convert_examples_to_features(examples, tokenizer, max_length):
    logger.info("正在创建 features")
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        inputs = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length,
                                       pad_to_max_length=True, truncation="longest_first")

        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]
        attention_mask = inputs['attention_mask']
        input_len, att_mask_len, token_type_len = len(input_ids), len(attention_mask), len(token_type_ids)
        assert input_len == max_length, "input_ids 长度错误 {} vs {}".format(input_len, max_length)
        assert att_mask_len == max_length, "att_mask 长度错误 {} vs {}".format(att_mask_len, max_length)
        assert token_type_len == max_length, "token_type_ids 长度错误 {} vs {}".format(token_type_len, max_length)

        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, label=example.label))
    return features


def load_and_cache_examples(args, processor, tokenizer, mode, examples=None):
    # if mode in ['train', 'dev', 'test']:
    assert mode == "train" and "dev" or "test", "mode 只支持 train|dev|test"

    # features 数据保存到本地文件
    if mode == 'train':
        cached_features_file = os.path.join(args.data_dir, 'cached_train_{}'.format(args.max_length))
    if mode == 'dev':
        cached_features_file = os.path.join(args.data_dir, 'cached_dev_{}'.format(args.max_length))
    if mode == 'test':
        cached_features_file = os.path.join(args.data_dir, 'cached_test_{}'.format(args.max_length))

    if os.path.exists(cached_features_file):
        logger.info("从本地文件加载 features，%s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("创建 features，%s", args.data_dir)
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir)
        if mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        if mode == 'test':
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(examples, tokenizer, args.max_length)
        logger.info("保存 features 到本地文件 %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(args.device)
    all_attention_mask = torch.LongTensor([f.attention_mask for f in features]).to(args.device)
    all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features]).to(args.device)
    all_labels = torch.LongTensor([f.label for f in features]).to(args.device)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
