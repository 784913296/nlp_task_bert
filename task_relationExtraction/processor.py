import os
import json
import logging
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor
from transformers import BertTokenizer
from task_relationExtraction.conf import args_relation_extraction
from task_relationExtraction.util import InputExample, InputFeatures, get_label2id

logger = logging.getLogger(__name__)


def convert_pos_to_mask(e_pos, max_length=128):
    e_pos_mask = [0] * max_length
    for i in range(e_pos[0], e_pos[1]):
        e_pos_mask[i] = 1
    return e_pos_mask


def get_pos(text, tokenizer, mask_entity=False):
    """ 获取分词后实体新的位置 """
    sentence = text['text']
    pos_head = text['h']['pos']
    pos_tail = text['t']['pos']
    if pos_head[0] > pos_tail[0]:
        pos_min = pos_tail
        pos_max = pos_head
        rev = True
    else:
        pos_min = pos_head
        pos_max = pos_tail
        rev = False

    sent0 = tokenizer.tokenize(sentence[:pos_min[0]])
    ent0 = tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
    sent1 = tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
    ent1 = tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
    sent2 = tokenizer.tokenize(sentence[pos_max[1]:])

    if rev:
        if mask_entity:
            ent0 = ['[unused6]']
            ent1 = ['[unused5]']
        pos_tail = [len(sent0), len(sent0) + len(ent0)]
        pos_head = [
            len(sent0) + len(ent0) + len(sent1),
            len(sent0) + len(ent0) + len(sent1) + len(ent1)
        ]
    else:
        if mask_entity:
            ent0 = ['[unused5]']
            ent1 = ['[unused6]']
        pos_head = [len(sent0), len(sent0) + len(ent0)]
        pos_tail = [
            len(sent0) + len(ent0) + len(sent1),
            len(sent0) + len(ent0) + len(sent1) + len(ent1)
        ]
    tokens = sent0 + ent0 + sent1 + ent1 + sent2

    re_tokens = ['[CLS]']
    cur_pos = 0
    pos1 = [0, 0]
    pos2 = [0, 0]
    for token in tokens:
        token = token.lower()
        if cur_pos == pos_head[0]:
            pos1[0] = len(re_tokens)
            re_tokens.append('[unused1]')
        if cur_pos == pos_tail[0]:
            pos2[0] = len(re_tokens)
            re_tokens.append('[unused2]')
        re_tokens.append(token)
        if cur_pos == pos_head[1] - 1:
            re_tokens.append('[unused3]')
            pos1[1] = len(re_tokens)
        if cur_pos == pos_tail[1] - 1:
            re_tokens.append('[unused4]')
            pos2[1] = len(re_tokens)
        cur_pos += 1
    re_tokens.append('[SEP]')
    return re_tokens[1:-1], pos1, pos2


class relationExProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"))

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"))

    def _create_examples(self, input_file):
        examples = []
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for i, line in tqdm(enumerate(f_in)):
                line = line.strip()
                text = json.loads(line)
                label = text['relation']
                examples.append(InputExample(guid=i, text=text, label=label))
        return examples



def convert_examples_to_features(examples, tokenizer, label2id, max_length=128):
    features = []
    for ex_index, example in enumerate(examples):
        text, pos_e1, pos_e2 = get_pos(example.text, tokenizer)
        tokenizer_inputs = tokenizer.encode_plus(text, add_special_tokens=True,
                                                 max_length=max_length,
                                                 pad_to_max_length=True,
                                                 truncation="longest_first")
        input_ids = tokenizer_inputs["input_ids"]
        token_type_ids = tokenizer_inputs["token_type_ids"]
        attention_mask = tokenizer_inputs["attention_mask"]
        label = label2id[example.label]
        e1_mask = convert_pos_to_mask(pos_e1, max_length)
        e2_mask = convert_pos_to_mask(pos_e2, max_length)

        assert len(input_ids) == max_length, f"错误的输入长度 {len(input_ids)} vs {max_length}"
        assert len(attention_mask) == max_length, f"错误的输入长度 {len(attention_mask)} vs {max_length}"
        assert len(token_type_ids) == max_length, f"错误的输入长度 {len(token_type_ids)} vs {max_length}"

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info(f"guid: {example.guid}")
            logger.info(f"input_ids: {input_ids}")
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info(f"label: {label}")
            logger.info(f"e1_mask: {e1_mask}")
            logger.info(f"e2_mask: {e2_mask}")
        features.append(InputFeatures(input_ids, attention_mask, token_type_ids,
                                      label, e1_mask, e2_mask))
    return features


def load_and_cache_example(args, tokenizer, processor, mode):
    assert mode == "train" or "dev" or "test", "mode 只支持 train|dev|test"
    cached_features_file = "cached_{}_{}".format(mode, str(args.max_length))
    cached_features_file = os.path.join(args.data_dir, cached_features_file)
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        if mode == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif mode == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif mode == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise ValueError("UNKNOW ERROR")

        features = convert_examples_to_features(examples, tokenizer, args.label2id, args.max_length)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(args.device)
    all_attention_mask = torch.LongTensor([f.attention_mask for f in features]).to(args.device)
    all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features]).to(args.device)
    all_label = torch.LongTensor([f.label for f in features]).to(args.device)
    all_e1_mask = torch.LongTensor([f.e1_mask for f in features]).to(args.device)
    all_e2_mask = torch.LongTensor([f.e2_mask for f in features]).to(args.device)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_label, all_e1_mask, all_e2_mask)
    return dataset


if __name__ == "——main__":
    args = args_relation_extraction()
    processor = relationExProcessor()
    train_examples = processor.get_train_examples(args.data_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    args.label2id = get_label2id(args.labels_file)
    # features = convert_examples_to_features(train_examples, tokenizer, args.label2id)
    train_dataset = load_and_cache_example(args, tokenizer, processor, mode="train")
