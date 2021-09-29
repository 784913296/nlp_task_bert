import argparse
import torch


def args_ner():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="voidful/albert_chinese_tiny", type=str,
                        help="voidful/albert_chinese_tiny|nghuyong/ernie-tiny")
    parser.add_argument("--data_dir", default="../data/task_ner", type=str, help="数据集文件目录")
    parser.add_argument("--output_dir", default="../model_data", type=str, help="模型输出目录")
    parser.add_argument("--bert_type", default="albert", type=str, help="albert|ernie")
    parser.add_argument("--task_type", default="task_ner", type=str, help="任务类型")
    parser.add_argument("--baseline", default=True, action="store_true", help="是否加载baseline目录的模型")
    parser.add_argument('--baseline_dir', default='../baseline', help='baseline 模型目录')

    parser.add_argument("--max_seq_length", default=128, type=int, help="输入到bert的最大长度，通常不应该超过512")
    parser.add_argument("--do_train", default=True, action='store_true', help="是否进行训练")
    parser.add_argument("--do_eval", default=True, action="store_true", help="是否进行验证")
    parser.add_argument("--do_pred", default=True, action="store_true", help="是否进行验证")
    parser.add_argument("--do_lower_case", default=False, action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="训练集的batch_size")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="验证集的batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="最大的梯度更新")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="epoch 数目")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--warmup_steps", default=0, type=int, help="让学习增加到1的步数，在warmup_steps后，再衰减到0")
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args
