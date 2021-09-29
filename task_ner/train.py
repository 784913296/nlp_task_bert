import logging
import warnings
from task_ner.processor import load_and_cache_example, NerProcessor
from utils.ner_utils import label2id_load
from task_ner.conf import args_ner
from models.model import create_model, create_tokenizer
from task_ner.trainer import train


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    warnings.filterwarnings("ignore")
    args = args_ner()
    args.label2id = label2id_load()
    args.id2label = {v: k for k, v in args.label2id.items()}
    args.num_labels = len(args.label2id)

    processor = NerProcessor()
    tokenizer = create_tokenizer(args)
    model_file = "../baseline/albert/task_ner/pytorch_model.bin"
    model = create_model(args, model_file)
    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_example(args, tokenizer, processor, 'train')
        eval_dataset = load_and_cache_example(args, tokenizer, processor, 'dev')
        train(args, train_dataset, eval_dataset, model)