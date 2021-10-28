import logging
from models.model import create_model, create_tokenizer
from task_relationExtraction.trainer import train, evaluate
from task_relationExtraction.conf import args_relationExtraction
from task_relationExtraction.util import get_id2label, get_label2id
from task_relationExtraction.processor import relationExProcessor, load_and_cache_example


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    args = args_relationExtraction()
    processor = relationExProcessor()

    args.label2id = get_label2id(args.labels_file)
    args.id2lable = get_id2label(args.labels_file)
    args.num_labels = len(args.id2lable)

    tokenizer = create_tokenizer(args)
    model_file = "../model_data/albert/task_relationExtraction/checkpoint-8600/pytorch_model.bin"
    model = create_model(args, model_file)
    model.to(args.device)

    train_dataset = load_and_cache_example(args, tokenizer, processor, mode="train")
    train(args, model, train_dataset)

    dev_dataset = load_and_cache_example(args, tokenizer, processor, mode="dev")
    print(evaluate(args, model, dev_dataset, args.id2lable))

