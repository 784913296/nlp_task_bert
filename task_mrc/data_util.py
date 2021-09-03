import os
import json
import logging
from io import open
import pandas as pd
from transformers.models.bert.tokenization_bert import whitespace_tokenize
logger = logging.getLogger(__name__)


class CreateDataFile:
    """ 将原始的数据集处理成 train.csv dev.csv test.csv """
    def __init__(self, args):
        self.args = args

    def get_train_file(self):
        data_file = os.path.join(self.args.data_dir, "cmrc2018_train.json")
        self.create_data_file(data_file, mode="train")

    def get_dev_file(self):
        data_file = os.path.join(self.args.data_dir, "cmrc2018_dev.json")
        self.create_data_file(data_file, mode="dev")

    def get_test_file(self):
        data_file = os.path.join(self.args.data_dir, "cmrc2018_trial.json")
        self.create_data_file(data_file, mode="test")

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def context_processor(self, context):
        """ 将 context 处理成两个列表：char 列表、char对应的id列表 """
        doc_tokens = []
        char_to_word_id = []
        for c in context:
            if self.is_whitespace(c):
                continue
            doc_tokens.append(c)
            char_to_word_id.append(len(doc_tokens) - 1)
        return doc_tokens, char_to_word_id

    def create_data_file(self, data_file, mode):
        assert mode == "train" and "dev" or "test", "mode 只支持 train|dev|test"
        if mode == "train":
            write_file = os.path.join(self.args.data_dir, "train.csv")
        elif mode == "dev":
            write_file = os.path.join(self.args.data_dir, "dev.csv")
        elif mode == "test":
            write_file = os.path.join(self.args.data_dir, "test.csv")

        with open(data_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]
            examples = []
            for entry in input_data:
                for paragraph in entry["paragraphs"]:
                    context = paragraph["context"]
                    doc_tokens, char_to_word_id = self.context_processor(context)

                    for qas in paragraph["qas"]:
                        qas_id = qas["id"]
                        question_text = qas["question"]
                        start_position = None
                        end_position = None
                        orig_answer_text = None
                        is_impossible = False

                        if mode == "train":
                            if len(qas["answers"]) != 1 and not is_impossible:
                                raise ValueError("对于训练模式，每个问题应有一个正确答案")
                            answer = qas["answers"][0]
                            answer_text = answer["text"]
                            answer_start = answer["answer_start"]
                            answer_length = len(answer_text)
                            context_length = len(char_to_word_id)
                            if answer_start > context_length - 1:
                                logger.warning("样本错误: answer_start %s vs context长度 %s", answer_start,
                                               context_length)
                                continue
                            # 答案在 context 里的开始与结束位置
                            start_position = char_to_word_id[answer_start]
                            end_position = answer_start + answer_length - 1
                            if end_position > context_length - 1:
                                logger.warning("样本错误: 结束位置 %s vs context长度 %s", end_position,
                                               context_length)
                                continue
                            end_position = char_to_word_id[answer_start + answer_length - 1]

                            # 只添加可以从文档中准确恢复文本的答案。如果这不能，可能是由于奇怪的Unicode，将跳过这个例子。
                            actual_text = "".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = "".join(whitespace_tokenize(answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                logger.warning("样本错误: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                                continue

                        doc_tokens = " ".join(doc_tokens)
                        example = [qas_id, question_text, doc_tokens, orig_answer_text, start_position,
                                   end_position, is_impossible]
                        examples.append(example)

            columns = ["qas_id", "question_text", "doc_tokens", "answer_text", "start_position",
                       "end_position", "is_impossible"]
            df = pd.DataFrame(examples, columns=columns)
            df.to_csv(write_file)
