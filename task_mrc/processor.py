import os
import logging
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor
from task_mrc.util import SquadExample, InputFeatures, _check_is_max_context,\
    get_doc_spans, get_position_start_end
from task_mrc.conf import args_mrc
from models.model import create_tokenizer

logger = logging.getLogger(__name__)


class mrcProcessor(DataProcessor):
    def __init__(self, args):
        self.args = args
    def get_train_examples(self):
        data_file = os.path.join(self.args.data_dir, 'train.csv')
        return self._create_examples(data_file)

    def get_dev_examples(self):
        data_file = os.path.join(self.args.data_dir, 'dev.csv')
        return self._create_examples(data_file)

    def get_test_examples(self):
        data_file = os.path.join(self.args.data_dir, 'test.csv')
        return self._create_examples(data_file)

    def _create_examples(self, data_file):
        examples = []
        columns = ["qas_id", "question_text", "context", "answer_text",
                   "answer_start", "is_impossible"]
        df = pd.read_csv(data_file, names=columns, header=0)
        for index, (qas_id, question_text, context, answer_text,
                    answer_start, is_impossible) in df.iterrows():
            example = SquadExample(qas_id=qas_id, question_text=question_text,
                                   context=context, answer_text=answer_text,
                                   answer_start=answer_start,
                                   is_impossible=is_impossible)
            examples.append(example)
        return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, mode):
    unique_id = 1000000000
    features = []
    feature_iterator = tqdm(examples, desc="正在生成{} features".format(mode))
    for (example_index, example) in enumerate(feature_iterator):
        question_text = tokenizer.tokenize(example.question_text)
        if len(question_text) > max_query_length:
            question_text = question_text[:max_query_length]

        context = example.context
        doc_tokens = tokenizer.tokenize(context)
        doc_tokens_index = []
        for idx, token in enumerate(doc_tokens):
            doc_tokens_index.append(idx)

        # 分词后新的开始跟结束位置
        tok_start_position, tok_end_position = get_position_start_end(context, example.answer_start,
                                                              example.answer_text, tokenizer,
                                                              mode, example.is_impossible)
        # 每段文档的长度 减去问题长度跟[CLS]、[SEP]、[SEP]
        max_paragraph_len = max_seq_length - len(question_text) - 3
        # 获取每个段的开始位置跟长度
        doc_spans = get_doc_spans(doc_tokens, max_paragraph_len, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            question_and_paragraphs = []
            token_to_orig_map = {}
            token_is_max_context = {}

            # p_mask: token大于1的掩码不能在答案中(token可以在答案中为0)
            # CLS p_mask 为0
            p_mask = [0]
            question_and_paragraphs.append("[CLS]")

            # 问题
            for token in question_text:
                p_mask.append(1)
                question_and_paragraphs.append(token)
            question_and_paragraphs.append("[SEP]")
            # [SEP] p_mask
            p_mask.append(1)

            # 内容段落
            paragraphs = []
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(question_and_paragraphs)] = doc_tokens_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(question_and_paragraphs)] = is_max_context
                paragraphs.append(doc_tokens[split_token_index])
                question_and_paragraphs.append(doc_tokens[split_token_index])
                p_mask.append(0)
            paragraph_len = doc_span.length

            # 结尾 [SEP]
            p_mask.append(1)
            question_and_paragraphs.append("[SEP]")
            inputs = tokenizer.encode_plus(question_text, paragraphs, add_special_tokens=True,
                                           max_length=max_seq_length, pad_to_max_length=True,
                                           truncation="longest_first")
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]

            while len(p_mask) < max_seq_length:
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(token_type_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            # 训练时，把不包含答案的段落丢弃（即 start_position end_position 都设为0）
            if mode == "train" and not span_is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                doc_offset = len(question_text) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset + 1

                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True

            cls_index = 0
            if mode == "train" and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 1:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % unique_id)
                logger.info("example_index: %s" % example_index)
                logger.info("doc_span_index: %s" % doc_span_index)
                logger.info("question_text: %s" % question_text)
                logger.info("paragraphs: %s" % paragraphs)
                logger.info("question_and_paragraphs: %s" % question_and_paragraphs)
                logger.info("token_to_orig_map: %s" % token_to_orig_map)
                logger.info("token_is_max_context: %s" % token_is_max_context)
                logger.info("input_ids: %s" % input_ids)
                logger.info("attention_mask: %s" % attention_mask)
                logger.info("token_type_ids: %s" % token_type_ids)
                logger.info("p_mask: %s" % p_mask)

                if mode == "train" and span_is_impossible:
                    logger.info("impossible example")

                if mode == "train" and not span_is_impossible:
                    answer_text = "".join(question_and_paragraphs[start_position:end_position])
                    logger.info("start_position: %d" % start_position)
                    logger.info("end_position: %d" % end_position)
                    logger.info("answer: %s" % answer_text)

                    # 只处理在内容里能还原答案的样本
                    if answer_text.find(example.answer_text) == -1:
                        logger.info("样本错误，答案不能在内容了里还原")
                        continue

            feature = InputFeatures(unique_id=unique_id,
                                    example_index=example_index,
                                    doc_span_index=doc_span_index,
                                    tokens=paragraphs,
                                    token_to_orig_map=token_to_orig_map,
                                    token_is_max_context=token_is_max_context,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    cls_index=cls_index,
                                    p_mask=p_mask,
                                    paragraph_len=paragraph_len,
                                    start_position=start_position,
                                    end_position=end_position,
                                    is_impossible=span_is_impossible)
            features.append(feature)
            unique_id += 1
    return features


def read_local_feature(args, mode):
    """ features 数据保存本地文件名 """
    if mode == 'train':
        cached_features_file = os.path.join(args.data_dir, 'cached_train_{}'.format(args.max_seq_length))
    if mode == 'dev':
        cached_features_file = os.path.join(args.data_dir, 'cached_dev_{}'.format(args.max_seq_length))
    if mode == 'test':
        cached_features_file = os.path.join(args.data_dir, 'cached_test_{}'.format(args.max_seq_length))
    return cached_features_file


def get_examples(processor, mode):
    if mode == 'train':
        examples = processor.get_train_examples()
    elif mode == 'dev':
        examples = processor.get_dev_examples()
    elif mode == 'test':
        examples = processor.get_test_examples()
    return examples


def load_and_cache_examples(args, tokenizer, mode):
    assert mode == "train" or "dev" or "test", "mode 只支持 train|dev|test"

    # 确保分布式训练中只有第一个进程处理数据集，其他进程将使用缓存
    if args.local_rank not in [-1, 0] and mode == "train":
        torch.distributed.barrier()

    # 加载本地缓存文件
    cached_features_file = read_local_feature(args, mode)
    if os.path.exists(cached_features_file):
        logger.info("从本地文件加载 features，%s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("创建 features")
        processor = mrcProcessor(args)
        examples = get_examples(processor, mode)
        features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                mode=mode)
        logger.info("保存 features 到本地文件 %s", cached_features_file)
        torch.save(features, cached_features_file)

    # 确保分布式训练中只有第一个进程处理数据集，其他进程将使用缓存
    if args.local_rank == 0 and mode == "train":
        torch.distributed.barrier()

    all_input_ids = torch.LongTensor([f.input_ids for f in features]).to(args.device)
    all_attention_mask = torch.LongTensor([f.attention_mask for f in features]).to(args.device)
    all_token_type_ids = torch.LongTensor([f.token_type_ids for f in features]).to(args.device)
    all_cls_index = torch.LongTensor([f.cls_index for f in features]).to(args.device)
    all_p_mask = torch.FloatTensor([f.p_mask for f in features]).to(args.device)

    if mode == "train":
        all_start_positions = torch.LongTensor([f.start_position for f in features]).to(args.device)
        all_end_positions = torch.LongTensor([f.end_position for f in features]).to(args.device)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_start_positions,
                                all_end_positions, all_cls_index, all_p_mask)
    else:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long).to(args.device)
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_example_index, all_cls_index, all_p_mask)
    return dataset


if __name__ == "__main__":
    args = args_mrc()
    logging.basicConfig(format='%(asctime)s - %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    processor = mrcProcessor(args)
    examples = processor.get_train_examples()
    tokenizer = create_tokenizer(args)
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            doc_stride=args.doc_stride,
                                            max_query_length=args.max_query_length,
                                            mode="train")
    print(features)

