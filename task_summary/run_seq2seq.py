# coding=utf-8
import os
import logging
import torch
import torch.distributed as dist
from task_summary import utils_seq2seq
from task_summary.conf import args_summary
from utils.util import set_seed
from task_summary.trainer import train, MODEL_CLASSES
from task_summary.processor import seq2seqProcessor, load_and_cache_examples


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


if __name__ == "__main__":
    args = args_summary()
    if args.local_rank == -1 or args.no_cuda:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        args.device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    set_seed(args.seed)
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.local_rank not in (-1, 0):
        # 确保在分布式培训只有第一个过程下载模型和词汇
        dist.barrier()
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.model_name_or_path, label_smoothing=args.label_smoothing,
                                          max_position_embeddings=args.max_position_embeddings)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.local_rank == 0:
        dist.barrier()

    if args.do_train:
        print("Loading Train Dataset", args.data_dir)
        # bi_uni_pipeline = [utils_seq2seq.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys()),
        #                                                     tokenizer.convert_tokens_to_ids, args.max_seq_length,
        #                                                     mask_source_words=False, skipgram_prb=args.skipgram_prb,
        #                                                     skipgram_size=args.skipgram_size,
        #                                                     mask_whole_word=args.mask_whole_word,
        #                                                     tokenizer=tokenizer)]
        #
        # file = os.path.join(args.data_dir, args.src_file if args.src_file else 'train.tgt')
        # train_dataset = utils_seq2seq.Seq2SeqDataset(file, args.train_batch_size, tokenizer,
        #                                              args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline)
        # train(args, train_dataset, config, n_gpu)

        processor = seq2seqProcessor()
        train_dataset = load_and_cache_examples(args, processor, tokenizer, mode="train")
        train(args, train_dataset, config, n_gpu)


