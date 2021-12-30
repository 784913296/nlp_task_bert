import os
import json
import logging
from random import randint, shuffle
from random import random as rand
import torch
from collections import defaultdict
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from tqdm import tqdm
from task_summary.trainer import logger


class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, masked_ids, masked_pos,
                 masked_weights, next_sentence_label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.masked_ids = masked_ids
        self.masked_pos = masked_pos
        self.masked_weights = masked_weights
        self.next_sentence_label = next_sentence_label


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    """ 截断超过 max_len-3 的句子 """
    if len(tokens_a) + len(tokens_b) > max_len-3:
        while len(tokens_a) + len(tokens_b) > max_len-3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b


class seq2seqProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train_data.json'))

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev.json'))

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test_data.json'))

    def _create_examples(self, data_file):
        examples = []
        with open(data_file, mode='r', encoding='utf8') as f:
            for id, line in tqdm(enumerate(f.readlines())):
                line = eval(line)
                if len(line) != 2:
                    continue

                src_text, tgt_text = line["src_text"], line["tgt_text"]
                example = InputExample(guid=id, text_a=src_text, text_b=tgt_text)
                examples.append(example)
        return examples


def convert_examples_to_features(args, examples, tokenizer, max_length, skipgram_prb=0,
                                 skipgram_size=0, mask_source_words=True):
    logger.info("正在创建 features")
    max_pred = args.max_pred
    mask_prob = args.mask_prob
    mask_whole_word = args.mask_whole_word

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        next_sentence_label = None
        indexer = tokenizer.convert_tokens_to_ids
        vocab_words = list(tokenizer.vocab.keys())
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)

        # -3  for special tokens [CLS], [SEP], [SEP]
        tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, max_length)
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        token_type_ids = [4] * (len(tokens_a) + 2) + [5] * (len(tokens_b) + 1)

        # 对于mask语言模型, 当序列较短时，预测数有时小于 max_pred
        effective_length = len(tokens_b)
        if mask_source_words:
            effective_length += len(tokens_a)
        n_pred = min(max_pred, max(1, int(round(effective_length * mask_prob))))

        # 候选的 masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a) + 2) and (tk != '[CLS]'):
                cand_pos.append(i)
            elif mask_source_words and (i < len(tokens_a) + 2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end
                while (new_st >= 0) and tokens[new_st].startswith('##'):
                    new_st -= 1
                while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                    new_end += 1
                return new_st, new_end

            if (skipgram_prb > 0) and (skipgram_size >= 2) and (rand() < skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, skipgram_size)
                if mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                # directly mask
                if mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        if len(masked_pos) > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:     # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:   # 50%
                tokens[pos] = get_random_word(vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1] * len(masked_tokens)

        masked_ids = indexer(masked_tokens)
        input_ids = indexer(tokens)
        padding = max_length - len(input_ids)
        input_ids.extend([0] * padding)
        token_type_ids.extend([0] * padding)

        _tril_matrix = torch.tril(torch.ones((max_length, max_length), dtype=torch.long))
        attention_mask = torch.zeros(max_length, max_length, dtype=torch.long)
        attention_mask = attention_mask[:, :len(tokens_a) + 2].fill_(1)
        second_st, second_end = len(tokens_a) + 2, len(tokens_a) + len(tokens_b) + 3
        attention_mask = attention_mask[second_st:second_end, second_st:second_end].copy_(
            _tril_matrix[:second_end - second_st, :second_end - second_st])

        # Zero Padding for masked target
        if max_pred > n_pred:
            padding = max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0] * padding)
            if masked_pos is not None:
                masked_pos.extend([0] * padding)
            if masked_weights is not None:
                masked_weights.extend([0] * padding)

        input_len, att_mask_len, token_type_len = len(input_ids), len(attention_mask), len(token_type_ids)
        assert input_len == max_length, "input_ids 长度错误 {} vs {}".format(input_len, max_length)
        assert att_mask_len == max_length, "att_mask 长度错误 {} vs {}".format(att_mask_len, max_length)
        assert token_type_len == max_length, "token_type_ids 长度错误 {} vs {}".format(token_type_len, max_length)
        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids,
                                      masked_ids=masked_ids,
                                      masked_pos=masked_pos,
                                      masked_weights=masked_weights,
                                      next_sentence_label=next_sentence_label))
    return features


def load_and_cache_examples(args, processor, tokenizer, mode):
    assert mode == "train" or "dev" or "test", "mode 只支持 train|dev|test"

    # features 数据保存到本地文件
    if mode == 'train':
        cached_features_file = os.path.join(args.data_dir, 'cached_train_{}'.format(args.max_length))
    if mode == 'dev':
        cached_features_file = os.path.join(args.data_dir, 'cached_dev_{}'.format(args.max_length))
    if mode == 'test':
        cached_features_file = os.path.join(args.data_dir, 'cached_test_{}'.format(args.max_length))

    examples = None
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

        features = convert_examples_to_features(args, examples, tokenizer, args.max_length, skipgram_prb=0,
                                                skipgram_size=0, mask_source_words=False)
        logger.info("保存 features 到本地文件 %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(args.device)
    all_attention_mask = [f.attention_mask for f in features]
    all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features]).to(args.device)
    all_masked_ids = torch.LongTensor([f.masked_ids for f in features]).to(args.device)
    all_masked_pos = torch.LongTensor([f.masked_pos for f in features]).to(args.device)
    all_masked_weights = torch.LongTensor([f.masked_weights for f in features]).to(args.device)
    all_next_sentence_label = [f.next_sentence_label for f in features]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_masked_ids, \
           all_masked_pos, all_masked_weights, all_next_sentence_label


if __name__ == "__main__":
    data_dir = "../data/task_summary"
    processor = seq2seqProcessor()
    examples = processor.get_train_examples(data_dir)

    from transformers import BertTokenizer
    from task_summary.conf import args_summary
    args = args_summary()
    tokenizer = BertTokenizer.from_pretrained("../baseline/unilm_chinese")


    # features = convert_examples_to_features(examples, tokenizer, max_length=128,
    #                                         skipgram_prb=0, skipgram_size=0,
    #                                         mask_source_words=False)
    # feature = features[0]
    # print(feature.input_ids)
    dataset = load_and_cache_examples(args, processor, tokenizer, mode="train")
    print(dataset)