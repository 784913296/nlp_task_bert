import logging
import warnings
import torch
from task_sim.processor import simProcessor, convert_one_example_to_features
from task_sim.conf import args_sim
from models.model import create_model, create_tokenizer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

args = args_sim()
processor = simProcessor()
args.num_labels = len(processor.get_labels())
tokenizer = create_tokenizer(args)


def pred(args, model, text_a, text_b):
    examples = processor._create_one_example(text_a, text_b)
    features = convert_one_example_to_features(examples, tokenizer, max_length=128)
    features = list(features)

    input_ids = torch.LongTensor([f.input_ids for f in features]).to(args.device)
    attention_mask = torch.LongTensor([f.attention_mask for f in features]).to(args.device)
    token_type_ids = torch.LongTensor([f.token_type_ids for f in features]).to(args.device)

    with torch.no_grad():
        inputs = {'input_ids': input_ids,
                  'attention_mask': attention_mask,
                  'token_type_ids': token_type_ids}
        outputs = model(**inputs)
        _, predict = outputs[0].max(1)
        predict = predict.item()
        return predict


if __name__ == "__main__":
    mode_file = "../model_data/albert/task_sim/checkpoint-15800/pytorch_model.bin"
    model = create_model(args, mode_file)
    model.to(args.device)

    if args.do_pred:
        # 相似
        # text_a = '看图猜一电影名'
        # text_b = '看图猜电影的名字！'

        # 不相似
        text_a = "昨天到的鸡蛋烂了一个弄的哪都是全拿去洗了"
        text_b = "我的鸡蛋收到了但有一个烂了"

        # id2lable = {0: '不相似', 1: '相似'}
        pred_lable = pred(args, model, text_a, text_b)
        print(pred_lable)