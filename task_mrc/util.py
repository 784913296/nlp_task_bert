# coding=utf-8
import logging
import collections


logger = logging.getLogger(__name__)
RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])


class SquadExample(object):
    """ 没有答案的 start_position end_position 为 -1 """
    def __init__(self, qas_id, question_text, context, answer_text=None,
                 answer_start=None, is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context = context
        self.answer_text = answer_text
        self.answer_start = answer_start
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", context: [%s]" % self.context
        if self.answer_start:
            s += ", start_position: %d" % self.answer_start
        if self.is_impossible:
            s += ", is_impossible: %r" % self.is_impossible
        return s


class InputFeatures(object):
    def __init__(self, unique_id, example_index, doc_span_index, tokens, token_to_orig_map,
                 token_is_max_context, input_ids, attention_mask, token_type_ids, cls_index, p_mask,
                 paragraph_len, start_position=None, end_position=None, is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class EvalOps(object):
    def __init__(self, data_file, pred_file, out_file="", na_prob_file="na_prob.json",
                 na_prob_thresh=1.0, out_image_dir=None, verbose=False):
        self.data_file = data_file
        self.pred_file = pred_file
        self.out_file = out_file
        self.na_prob_file = na_prob_file
        self.na_prob_thresh = na_prob_thresh
        self.out_image_dir = out_image_dir
        self.verbose = verbose


def get_position_start_end(context, answer_start, answer_text, tokenizer, mode, is_impossible):
    """ 重新处理开始与结束位置，使用Bert分词，要重新标注起始和结束的位置 """
    position_start = None
    position_end = None

    if mode == "train" and is_impossible:
        position_start = -1
        position_end = -1

    if mode == "train" and not is_impossible:
        # 分词后，新的开始位置
        before_context = context[:answer_start]
        position_start = len(tokenizer.tokenize(before_context))
        position_end = len(tokenizer.tokenize(answer_text)) + position_start - 1
        if position_end > len(context) - 1:
            position_end = len(context) - 1
        # 标记化答案
        # position_start, position_end = _improve_answer_span(
        #     context, position_start, position_end, tokenizer, answer_text)
    return position_start, position_end


def get_doc_spans(doc_tokens, max_paragraph_len, doc_stride):
    """ 内容分段，采用了滑动窗口方法获取每个段落的开始位置跟长度
    doc_stride：每段的最大长度
    """
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(doc_tokens):
        length = len(doc_tokens) - start_offset
        if length > max_paragraph_len:
            length = max_paragraph_len
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(doc_tokens):
            break
        start_offset += min(length, doc_stride)
    return doc_spans


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, answer_text):
    """ 返回与带注释的答案更好匹配的标记化答案范围
        问题: 约翰·史密斯哪一年出生?
        背景: 领袖是约翰·史密斯(John Smith, 1895-1943)。
        回答: 1895
        标记化后答案: 1895年 """
    tok_answer_text = " ".join(tokenizer.tokenize(answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end
    return input_start, input_end


def _check_is_max_context(doc_spans, cur_span_index, position):
    """ 检查这是词是否“最大上下文”文档 """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

