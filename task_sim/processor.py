# coding=utf-8
from __future__ import absolute_import, division, print_function
import os, logging
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class simProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train.txt'))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'))

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test.tsv'))

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, path):
        logger.info("创建 examples 中")
        lines = open(path, 'r', encoding="utf-8").readlines()
        for id, line in enumerate(lines):
            text_a, text_b, label = line.strip().split('\t')
            yield InputExample(guid=id, text_a=text_a, text_b=text_b, label=label)

    def _create_one_example(self, text_a, text_b):
        yield InputExample(guid=1, text_a=str(text_a), text_b=str(text_b), label=None)


def convert_examples_to_features(examples, tokenizer, max_length=128, label_list=None):
    logger.info("创建 dataset 中")
    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,
                                       pad_to_max_length=True, truncation="longest_first")
        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs['attention_mask']
        input_len, att_mask_len, token_type_len = len(input_ids), len(attention_mask), len(token_type_ids)
        assert input_len == max_length, "input_ids 长度错误 {} vs {}".format(input_len, max_length)
        assert att_mask_len == max_length, "att_mask 长度错误 {} vs {}".format(att_mask_len, max_length)
        assert token_type_len == max_length, "token_type_ids 长度错误 {} vs {}".format(token_type_len, max_length)

        label_map = {label: i for i, label in enumerate(label_list)}
        label = label_map[int(example.label)]


        if ex_index < 3:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, label=label))
    return features


def convert_one_example_to_features(examples, tokenizer, max_length=128):
    features = []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,
                                       pad_to_max_length=True, runcation="longest_first")

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = inputs["attention_mask"]
        input_len, att_mask_len, token_type_len = len(input_ids), len(attention_mask), len(token_type_ids)
        assert input_len == max_length, "input_ids 长度错误 {} vs {}".format(input_len, max_length)
        assert att_mask_len == max_length, "att_mask 长度错误 {} vs {}".format(att_mask_len, max_length)
        assert token_type_len == max_length, "token_type_ids 长度错误 {} vs {}".format(token_type_len, max_length)
        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids, label=None))
    return features


def load_and_cache_examples(args, processor, tokenizer, evaluate=False):

    # features 数据保存到本地文件
    if evaluate:
        cached_features_file = os.path.join(args.data_dir, 'cached_dev_{}'.format(str(args.max_length)))
    else:
        cached_features_file = os.path.join(args.data_dir, 'cached_train_{}'.format(str(args.max_length)))

    if os.path.exists(cached_features_file):
        logger.info("从本地文件加载 features，%s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("创建本地文件保存 features，%s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(examples, tokenizer, args.max_length, label_list)
        logger.info("保存 features 到本地文件 %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.LongTensor([f.input_ids for f in features])
    all_attention_mask = torch.LongTensor([f.attention_mask for f in features])
    all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features])
    all_labels = torch.LongTensor([f.label for f in features])
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

