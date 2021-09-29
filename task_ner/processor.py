import os
import logging
import codecs
import torch
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor
from utils.ner_utils import bio_to_bioes
from tqdm import tqdm


logger = logging.getLogger(__name__)


class CrfInputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class CrfInputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


# 去掉标签的填充部分
def real_labels(attention_mask: torch.Tensor, labels: list):
    assert len(attention_mask) == len(labels)
    for i in range(len(attention_mask)):
        mask = attention_mask[i].ge(1)
        real_len = len(torch.masked_select(attention_mask[i], mask).tolist())
        label = labels[i]
        real_label = label[:real_len]
        yield real_label


class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "train.txt"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "dev.txt"))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, "test.txt"))

    def get_labels(self, args):
        return [key for key in args.label2id.keys()]

    @classmethod
    def _create_examples(cls, path):
        lines = []
        max_len = 0
        with codecs.open(path, 'r', encoding='utf-8') as f:
            word_list = []
            label_list = []
            for line in f:
                tokens = line.strip().split(' ')
                if 2 == len(tokens):
                    word = tokens[0]
                    label = tokens[1]
                    word_list.append(word)
                    label_list.append(label)
                elif 1 == len(tokens) and '' == tokens[0]:
                    if len(label_list) > max_len:
                        max_len = len(label_list)
                    lines.append((word_list, label_list))
                    word_list = []
                    label_list = []
        examples = []
        for i, (sentence, label) in enumerate(lines):
            label = bio_to_bioes(label)
            examples.append(CrfInputExample(guid=i, text=" ".join(sentence), label=label))

        return examples


def crf_convert_examples_to_features(args, examples, tokenizer, max_length=256, label_list=None):
    logger.info("正在生成 features")
    label2id = args.label2id
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Iteration")):
        tokenizer_inputs = tokenizer.encode_plus(example.text, add_special_tokens=True, max_length=max_length,
                                       pad_to_max_length=True, truncation="longest_first")
        input_ids = tokenizer_inputs["input_ids"]
        token_type_ids = tokenizer_inputs["token_type_ids"]
        attention_mask = tokenizer_inputs["attention_mask"]

        if label_list:
            # 加上 [START] 和  [END],  label2id["O"]*padding_length是[pad] ，把这些都暂时算作"O"，后面用 mask 来消除这些
            padding_length = max_length - len(example.label) - 2
            labels_ids = [label2id["[START]"]] + [label2id[l] for l in example.label] \
                         + [label2id["[END]"]] + [label2id["O"]] * padding_length
            assert len(labels_ids) == max_length, "Error with input length {} vs {}".format(len(labels_ids), max_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s " % " ".join([str(x) for x in labels_ids]))

        features.append(CrfInputFeatures(input_ids, attention_mask, token_type_ids, labels_ids))
    return features


def load_and_cache_example(args, tokenizer, processor, mode):
    assert mode == "train" or "dev" or "test", "mode 只支持 train|dev|test"
    cached_features_file = "cached_{}_{}".format(mode, str(args.max_seq_length))
    cached_features_file = os.path.join(args.data_dir, cached_features_file)
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels(args)
        if mode == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif mode == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise ValueError("UNKNOW ERROR")

        features = crf_convert_examples_to_features(args, examples, tokenizer, args.max_seq_length, label_list)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
    return dataset