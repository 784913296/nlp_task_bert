# coding=utf-8
import os
import json
import logging
from io import open
import pandas as pd
from models.model import create_tokenizer
from task_mrc.conf import args_mrc

logger = logging.getLogger(__name__)


class CreateDataFile:
    """ 将原始的数据集处理成 train.csv dev.csv test.csv """
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def get_train_file(self):
        data_file = os.path.join(self.args.data_dir, "cmrc2018_train.json")
        self.create_data_file(data_file, mode="train")

    def get_dev_file(self):
        data_file = os.path.join(self.args.data_dir, "cmrc2018_dev.json")
        self.create_data_file(data_file, mode="dev")

    def get_test_file(self):
        data_file = os.path.join(self.args.data_dir, "cmrc2018_trial.json")
        self.create_data_file(data_file, mode="test")

    def create_data_file(self, data_file, mode):
        assert mode == "train" and "dev" or "test", "mode 只支持 train|dev|test"
        if mode == "train":
            write_file = os.path.join(self.args.data_dir, "train.csv")
        elif mode == "dev":
            write_file = os.path.join(self.args.data_dir, "dev.csv")
        elif mode == "test":
            write_file = os.path.join(self.args.data_dir, "test.csv")

        with open(data_file, "r", encoding='utf-8') as f:
            input_data = json.load(f)["data"]
            examples = []
            for entry in input_data:
                for paragraph in entry["paragraphs"]:
                    context = paragraph["context"]
                    if not context:
                        context = "没有内容"

                    for qas in paragraph["qas"]:
                        qas_id = qas["id"]
                        question_text = qas["question"]
                        if not question_text:
                            question_text = "没有问题"
                        answer_start = None
                        answer_text = None
                        is_impossible = False

                        if mode == "train":
                            if len(qas["answers"]) != 1 and not is_impossible:
                                raise ValueError("对于训练模式，每个问题应有一个正确答案")
                            answer = qas["answers"][0]
                            answer_text = answer["text"]
                            answer_start = answer["answer_start"]

                            # 只保存在内容里能还原答案的样本
                            answer_text_len = len(answer_text)
                            context_answer = context[answer_start: answer_start + answer_text_len]
                            if context_answer.find(answer_text) == -1:
                                logger.info("样本错误，答案不能在内容了里还原")
                                continue

                        example = [qas_id, question_text, context, answer_text, answer_start,
                                   is_impossible]
                        examples.append(example)

            columns = ["qas_id", "question_text", "context", "answer_text", "answer_start", "is_impossible"]
            df = pd.DataFrame(examples, columns=columns)
            df.to_csv(write_file)


if __name__ == "__main__":
    args = args_mrc()
    args.num_labels = 2
    tokenizer = create_tokenizer(args)
    create_file = CreateDataFile(args, tokenizer)
    create_file.get_train_file()
    create_file.get_dev_file()
    create_file.get_test_file()
