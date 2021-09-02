# coding:utf-8
import argparse
import torch


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()
        return parser

    @staticmethod
    def initialize(parser: argparse.ArgumentParser):
        parser.add_argument("--model_name", default="voidful/albert_chinese_tiny", type=str,
                            help="voidful/albert_chinese_tiny|nghuyong/ernie-tiny")
        parser.add_argument('--output_dir', default='../model_data', help='the output dir for model checkpoints')
        parser.add_argument("--bert_type", default="albert", type=str, help="albert|ernie")
        parser.add_argument("--task_type", default="task_classes", type=str, help="任务类型")
        parser.add_argument("--baseline", default=False, action="store_true", help="是否加载baseline目录的模型")
        parser.add_argument('--baseline_dir', default='../baseline', help='baseline 模型目录')
        parser.add_argument("--num_labels", default=327, help="分类个数")

        parser.add_argument('--data_dir', default='../data/task_classes', help='数据集目录')
        parser.add_argument('--data_file', default='../data/task_classes/train_data', help='数据集')
        parser.add_argument("--cid2index_file", default="../data/task_classes/cid2index.txt", help="")
        parser.add_argument("--cid2name_file", default="../data/task_classes/cid2name.txt", help="")
        parser.add_argument("--index2cid_file", default="../data/task_classes/index2cid.txt", help="")

        parser.add_argument('--seed', type=int, default=42, help="random seed")
        parser.add_argument("--max_length", default=64, type=int, help="max_length")
        parser.add_argument("--evaluate_during_training", action='store_true', help="在每个测井步骤的培训期间进行Rul评估。")
        parser.add_argument("--train_batch_size", default=256, type=int, help="train_batch_size")
        parser.add_argument("--eval_batch_size", default=128, type=int, help="eval_batch_size")
        parser.add_argument("--do_train", default=True, action='store_true', help="是否进行训练")
        parser.add_argument("--do_eval", default=True, action="store_true", help="是否进行验证")
        parser.add_argument("--do_stack", default=True, action="store_true", help="是否进行验证")
        parser.add_argument("--num_train_epochs", default=20, type=int, help="train epochs")
        parser.add_argument('--save_steps', type=int, default=200, help="多少个step保存一次模型")
        parser.add_argument('--swa_start', default=3, type=int, help='the epoch when swa start')

        parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="在执行发现传播之前要累积的更新步骤数")
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="learning_rate")
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Adam 的 Epsilon")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument("--max_steps", default=-1, type=int, help="要训练的步骤总数，会覆盖 num_train_epochs")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument('--logging_steps', type=int, default=10, help="logging steps")
        parser.add_argument("--do_lower_case", default=False, action="store_true",
                            help="Set this flag if you are using an uncased model.")
        return parser

    def get_parser(self):
        parser = self.parse()
        parser = self.initialize(parser)
        args = parser.parse_args()
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return args
