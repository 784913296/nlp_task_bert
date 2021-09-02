import json
import logging
from io import open
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from task_mrc.util import SquadExample

logger = logging.getLogger(__name__)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def create_data_file(data_file, is_training):
    with open(data_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                for c in paragraph_text:
                    if is_whitespace(c):
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
                            logger.warning("样本错误: '%s'  offfset vs. length'%s'", answer_offset,
                                           len(char_to_word_offset))
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
                            logger.warning("样本错误: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                            continue

                    example = SquadExample(qas_id=qas_id, question_text=question_text, doc_tokens=doc_tokens,
                                           orig_answer_text=orig_answer_text, start_position=start_position,
                                           end_position=end_position, is_impossible=is_impossible)
                    examples.append(example)
        return examples

