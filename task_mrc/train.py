# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import logging
import torch
from task_mrc.processor import load_and_cache_examples
from task_mrc.conf import args_mrc
from task_mrc.trainer import train, evaluate
from models.model import create_model, create_tokenizer
from task_mrc.data_util import CreateDataFile


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    args = args_mrc()
    args.num_labels = 2

    # 原始的数据集处理成 train.csv dev.csv test.csv
    create_file = CreateDataFile(args)
    create_file.get_train_file()
    create_file.get_dev_file()
    create_file.get_test_file()

    # # 设置远程调试
    # if args.server_ip and args.server_port:
    #     import ptvsd
    #     print("Waiting for debugger attach")
    #     ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    #     ptvsd.wait_for_attach()
    #
    # # 设置 CUDA GPU 和分布式训练
    # if args.local_rank == -1 or args.no_cuda:
    #     device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # else:
    #     # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    #     torch.distributed.init_process_group(backend='nccl')
    #     args.n_gpu = 1
    # args.device = device
    #
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #                 args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)
    #
    # # 确保分布式训练中的第一个过程才会下载model跟vocab
    # if args.local_rank not in [-1, 0]:
    #     torch.distributed.barrier()
    #
    # tokenizer = create_tokenizer(args)
    # model_file = ""
    # model = create_model(args, model_file)
    # model.to(args.device)
    #
    # if args.do_train:
    #     train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    #     global_step, tr_loss = train(args, train_dataset, model, tokenizer)
    #     logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    #
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     result = evaluate(args, model, tokenizer)
    #     print(result)


