import json
import torch
import argparse
from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer, BertForQuestionAnswering,\
    AlbertConfig, AlbertTokenizer, AlbertForQuestionAnswering
from task_mrc.trainer import evaluate
import os

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'albert': (AlbertConfig, AlbertForQuestionAnswering, BertTokenizer)
}

parser = argparse.ArgumentParser()
parser.add_argument("--train_file", default='./data/cmrc2018_train.json', type=str, help="训练数据集")
parser.add_argument("--predict_file", default='./data/cmrc2018_dev.json', type=str, help="验证数据集")
parser.add_argument("--model_type", default="albert", type=str, help="bert、albert")
parser.add_argument("--model_name", default='voidful/albert_chinese_tiny', type=str, help="model name")
parser.add_argument("--output_dir", default='./model_data/', type=str, help="模型输出目录")
parser.add_argument("--cache_dir", default="", type=str, help="从s3下载的模型保存目录")
parser.add_argument("--do_train", action='store_true', default=True, help="训练模式")
parser.add_argument("--do_eval", action='store_true', default=True, help="验证模式")
parser.add_argument('--overwrite_output_dir', action='store_true', help="覆盖 output_dir")
parser.add_argument('--overwrite_cache', action='store_true', help="覆盖 cache")
parser.add_argument("--mode_filename", default='./model_data/pytorch_model.bin', type=str, help="预训练后的模型")

parser.add_argument("--max_seq_length", default=128, type=int, help="句子最大长度")
parser.add_argument("--max_query_length", default=64, type=int, help="问题的最大长度")
parser.add_argument("--max_answer_length", default=30, type=int, help="答案的最大长度")
parser.add_argument("--doc_stride", default=128, type=int, help="长文档分割成多个块时，最大的长度")
parser.add_argument("--per_gpu_train_batch_size", default=128, type=int, help="train_batch_size")
parser.add_argument("--per_gpu_eval_batch_size", default=128, type=int, help="eval_batch_size")

parser.add_argument('--save_steps', type=int, default=100, help="保存模型的的步数")
parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")

# warmup
parser.add_argument("--num_train_epochs", default=1.0, type=float, help="训练次数")
parser.add_argument("--max_steps", default=-1, type=int, help="训练总次数，会覆盖 num_train_epochs")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="执行向后/更新传递之前要累积的更新步骤数")

parser.add_argument("--n_best_size", default=20, type=int, help="最佳预测的个数，输出到 nbest_predictions.json")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

# GPU 相关参数
parser.add_argument("--no_cuda", action='store_true', help="有CUDA时是否使用")
parser.add_argument("--local_rank", type=int, default=-1, help="local_rank 用于gpu上的分布式训练")
parser.add_argument('--fp16', action='store_true', help="是否使用16位(混合)精度(通过NVIDIA apex)而不是32位精度")
parser.add_argument('--fp16_opt_level', type=str, default='O1', help="Apex AMP优化级别 [O0, O1, O2, O3]")

parser.add_argument('--version_2_with_negative', action='store_true', help='如果为真，队列的例子中有一些没有答案。')
parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                    help="null_score - best_non_null 大于阈值，则预测null")
parser.add_argument("--do_lower_case", action='store_true', help="使用未加区分大小写的模型，要设置此标志")
parser.add_argument("--evaluate_during_training", action='store_true', help="每个训练步骤进行 Rul evaluation")
parser.add_argument("--verbose_logging", action='store_true', help="True 打印所有与数据处理相关的警告")
parser.add_argument("--eval_all_checkpoints", action='store_true', help="计算所有以与model_name结尾相同的前缀开始并以步骤号结束的检查点")

args = parser.parse_args()
args.n_gpu = torch.cuda.device_count()
args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.model_name, do_lower_case=False)
config = config_class.from_pretrained(args.model_name)
model = model_class(config)
model.load_state_dict(torch.load(args.mode_filename))
model.to(args.device)
model.eval()
input_file = args.predict_file


def handle_file(input_file, context, question):
    with open(input_file, "r", encoding="utf-8") as reader:
        orig_data = json.load(reader)
        orig_data["data"][0]['paragraphs'][0]['context'] = context
        for i in range(len(question)):
            orig_data["data"][0]['paragraphs'][0]['qas'][i][
                'question'] = question[i]
    with open(input_file, "w") as writer:
        writer.write(json.dumps(orig_data, indent=4) + "\n")


def run():
    while True:
        raw_text = input("Please Enter:")
        while not raw_text:
            print('Input should not be empty!')
            raw_text = input("Please Enter:")
        context = ''
        question = []
        try:
            raw_json = json.loads(raw_text)
            context = raw_json['context']
            if not context:
                continue
            raw_qas = raw_json['qas']
            if not raw_qas:
                continue
            for i in range(len(raw_qas)):
                question.append(raw_qas[i]['question'])
        except Exception as identifier:
            print(identifier)
            continue
        handle_file(input_file, context, question)
        # evaluate(args, model, tokenizer)

        predict_file = os.path.join(args.output_dir, "predictions_.json")
        with open(predict_file, "r", encoding="utf-8") as reader:
            orig_data = json.load(reader)
            print(orig_data["DEV_0_QUERY_0"])
        # clean input file
        handle_file(input_file, "", ["", "", ""])


if __name__ == "__main__":
    print(run())
