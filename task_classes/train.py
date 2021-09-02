# coding:utf-8
import os
import logging
from task_classes.processor import classesProcessor, load_and_cache_examples
from task_classes.conf import Args
from task_classes.trainer import train, evaluate, stacking
from models.model import create_model, create_tokenizer


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

    args = Args().get_parser()
    processor = classesProcessor()
    tokenizer = create_tokenizer(args)

    model_file = os.path.join(args.output_dir, args.bert_type, args.task_type,
                              'checkpoint-1600', 'pytorch_model.bin')
    model = create_model(args, model_file)
    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, processor, tokenizer, mode="train")
        train_loss = train(args, model, processor, tokenizer, train_dataset)
        logging.info("训练结束：loss {}".format(train_loss))

    if args.do_eval:
        eval_dataset = load_and_cache_examples(args, processor, tokenizer, mode="dev")
        eval = evaluate(args, model, eval_dataset)
        logging.info("验证结束：{}".format(eval))

    if args.do_stack:
        stacking(args, processor, tokenizer, model)
