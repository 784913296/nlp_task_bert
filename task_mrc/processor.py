from __future__ import absolute_import, division, print_function
import os
import json
import logging
import collections
from io import open
import torch
from torch.utils.data import TensorDataset
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from transformers.data.processors.utils import DataProcessor
from task_mrc.util import SquadExample, InputFeatures, _improve_answer_span, _check_is_max_context

logger = logging.getLogger(__name__)


class mcrProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'cmrc2018_train.json'), is_training=True)

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'cmrc2018_dev.json'))

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _create_examples(self, data_file, is_training=False):
        with open(data_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
            examples = []
            for entry in input_data:
                for paragraph in entry["paragraphs"]:
                    paragraph_text = paragraph["context"]
                    doc_tokens = []
                    char_to_word_offset = []
                    for c in paragraph_text:
                        if self.is_whitespace(c):
                            continue
                        doc_tokens.append(c)
                        char_to_word_offset.append(len(doc_tokens) - 1)

                    for qa in paragraph["qas"]:
                        qas_id = qa["id"]
                        question_text = qa["question"]
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                        is_impossible = False
                        if is_training:
                            if (len(qa["answers"]) != 1) and (not is_impossible):
                                raise ValueError("对于训练模式，每个问题应有一个正确答案")
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            if answer_offset > len(char_to_word_offset) - 1:
                                logger.warning("样本错误: '%s'  offfset vs. length'%s'", answer_offset, len(char_to_word_offset))
                                continue
                            start_position = char_to_word_offset[answer_offset]
                            end_position = answer_offset + answer_length - 1
                            if end_position > len(char_to_word_offset) - 1:
                                logger.warning("样本错误: '%s' vs. '%s'", end_position, len(char_to_word_offset))
                                continue
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]

                            # 只添加可以从文档中准确恢复文本的答案。如果这不能，可能是由于奇怪的Unicode，将跳过这个例子。
                            # 注意，对于训练模式，不能保证每个示例都是保留的
                            actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = "".join(whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("样本错误: '%s' vs. '%s'", actual_text,cleaned_answer_text)
                                continue

                        example = SquadExample(qas_id=qas_id, question_text=question_text, doc_tokens=doc_tokens,
                                               orig_answer_text=orig_answer_text, start_position=start_position,
                                               end_position=end_position, is_impossible=is_impossible)
                        examples.append(example)
            return examples


# data_dir = "../data/task_mrc"
# version_2_with_negative = 0.0
# processor = mcrProcessor()
# examples = processor.get_train_examples(data_dir)
# print(examples[0])


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training=False,
                                 cls_token_at_end=False, cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=0,
                                 pad_token_segment_id=0, mask_padding_with_zero=True):
    unique_id = 1000000000
    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index, orig_to_tok_index, all_doc_tokens = [], [], []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position, tok_end_position = None, None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])   # pylint: disable=invalid-name
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(
                    doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join(["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))

            feature = InputFeatures(unique_id=unique_id, example_index=example_index, doc_span_index=doc_span_index,
                                    tokens=tokens, token_to_orig_map=token_to_orig_map, token_is_max_context=token_is_max_context,
                                    input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, cls_index=cls_index,
                                    p_mask=p_mask, paragraph_len=paragraph_len, start_position=start_position,
                                    end_position=end_position, is_impossible=span_is_impossible)
            features.append(feature)
            unique_id += 1

    return features


# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_seq_length=128,
#                                         doc_stride=128, max_query_length=64, is_training=False)
# print(features)

def load_and_cache_examples(args, tokenizer, mode, output_examples=False):
    assert mode == "train" and "dev" or "test", "mode 只支持 train|dev|test"

    # 确保分布式训练中只有第一个进程处理数据集，其他进程将使用缓存
    if args.local_rank not in [-1, 0] and mode == "train":
        torch.distributed.barrier()

    # features 数据保存到本地文件
    if mode == 'train':
        cached_features_file = os.path.join(args.data_dir, 'cached_train_{}'.format(args.max_seq_length))
    if mode == 'dev':
        cached_features_file = os.path.join(args.data_dir, 'cached_dev_{}'.format(args.max_seq_length))
    if mode == 'test':
        cached_features_file = os.path.join(args.data_dir, 'cached_test_{}'.format(args.max_seq_length))

    if os.path.exists(cached_features_file):
        logger.info("从本地文件加载 features，%s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("创建 features，%s", args.data_dir)
        processor = mcrProcessor()
        if mode == 'train':
            examples = processor.get_train_examples(args.data_dir)
            is_training = True
        if mode == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
            is_training = False
        if mode == 'test':
            examples = processor.get_test_examples(args.data_dir)
            is_training = False

        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=is_training)
        logger.info("保存 features 到本地文件 %s", cached_features_file)
        torch.save(features, cached_features_file)

    # 确保分布式训练中只有第一个进程处理数据集，其他进程将使用缓存
    if args.local_rank == 0 and mode == "train":
        torch.distributed.barrier()

    all_input_ids = torch.LongTensor([f.input_ids for f in features])
    all_input_mask = torch.LongTensor([f.input_mask for f in features])
    all_segment_ids = torch.LongTensor([f.segment_ids for f in features])
    all_cls_index = torch.LongTensor([f.cls_index for f in features])
    all_p_mask = torch.FloatTensor([f.p_mask for f in features])

    if mode == "train":
        all_start_positions = torch.LongTensor([f.start_position for f in features])
        all_end_positions = torch.LongTensor([f.end_position for f in features])
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions,
                                all_end_positions, all_cls_index, all_p_mask)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset
