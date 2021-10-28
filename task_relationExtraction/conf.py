import argparse
import torch


def args_relationExtraction():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="voidful/albert_chinese_tiny", type=str,
                        help="voidful/albert_chinese_tiny | nghuyong/ernie-tiny")
    parser.add_argument('--output_dir', default='../model_data', help='the output dir for model checkpoints')
    parser.add_argument("--bert_type", default="albert", type=str, help="albert|ernie")
    parser.add_argument("--task_type", default="task_relationExtraction", type=str, help="任务类型")
    parser.add_argument("--baseline", default=False, action="store_true", help="是否加载baseline目录的模型")
    parser.add_argument('--baseline_dir', default='../baseline', help='baseline 模型目录')
    parser.add_argument('--data_dir', default='../data/task_relationExtraction', help='数据集目录')
    parser.add_argument('--train_file', default='../data/task_relationExtraction/train.jsonl',
                        help='训练数据集')
    parser.add_argument('--dev_file', default='../data/task_relationExtraction/val.jsonl',
                        help='验证数据集')
    parser.add_argument("--labels_file", type=str, default="../data/task_relationExtraction/relation.txt")

    # 训练参数
    parser.add_argument("--num_train_epochs", default=20, type=int, help="epoch 数目")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=200, help="保存检查点的步数")
    parser.add_argument('--dropout', type=float, default=0.1, required=False, help='dropout')
    parser.add_argument("--seed", type=int, default=42)

    # optimizer
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="权重衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="让学习增加到1的步数，在warmup_steps后，再衰减到0")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,help="梯度累计更新的步骤，用来弥补GPU过小的情况")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="最大的梯度更新")

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args
