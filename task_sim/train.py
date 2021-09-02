import logging
import warnings
from task_sim.processor import simProcessor, load_and_cache_examples
from task_sim.conf import args_sim
from task_sim.trainer import train, evaluate
from models.model import create_model, create_tokenizer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args = args_sim()
    processor = simProcessor()
    args.num_labels = len(processor.get_labels())
    tokenizer = create_tokenizer(args)

    mode_file = "../model_data/albert/task_sim/checkpoint-15800/pytorch_model.bin"
    model = create_model(args, mode_file)
    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, processor, tokenizer, evaluate=False)
        train_ = train(args, processor, model, tokenizer, train_dataset)
        print(train_)

    if args.do_eval:
        eval_dataset = load_and_cache_examples(args, processor, tokenizer, evaluate=True)
        eval = evaluate(args, model, eval_dataset)
        print(eval)