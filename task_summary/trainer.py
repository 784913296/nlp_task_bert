import os
import glob
from pathlib import Path
import logging
import math
from tqdm import tqdm, trange
import torch
import torch.distributed as dist
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup
from task_summary.modeling_unilm import UnilmForSeq2Seq
from task_summary import utils_seq2seq


MODEL_CLASSES = {'unilm': (BertConfig, UnilmForSeq2Seq, BertTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)



def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def train(args, train_dataset, config, n_gpu):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    device = args.device
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset, replacement=False)
        _batch_size = args.train_batch_size
    else:
        train_sampler = DistributedSampler(train_dataset)
        _batch_size = args.train_batch_size // dist.get_world_size()
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
    #                                                num_workers=args.num_workers,
    #                                                collate_fn=utils_seq2seq.batch_list_to_batch_tensors,
    #                                                pin_memory=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                   num_workers=args.num_workers)
    t_total = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    global_step = 0
    if (recover_step is None) and (args.model_recover_path is None):
        model_recover = None
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(
                args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
            # recover_step == number of epochs
            global_step = math.floor(
                recover_step * t_total / args.num_train_epochs)
        elif args.model_recover_path:
            logger.info("***** Recover model: %s *****", args.model_recover_path)
            model_recover = torch.load(args.model_recover_path, map_location='cpu')
    model = model_class.from_pretrained(args.model_name_or_path, state_dict=model_recover, config=config)
    if args.local_rank == 0:
        dist.barrier()

    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*t_total),
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model, device_ids=[
                    args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)

        logger.info("***** Recover amp: %d *****", recover_step)
        amp_recover = torch.load(os.path.join(
            args.output_dir, "amp.{0}.bin".format(recover_step)), map_location='cpu')
        amp.load_state_dict(amp_recover)

        logger.info("***** Recover scheduler: %d *****", recover_step)
        scheduler_recover = torch.load(os.path.join(
            args.output_dir, "sched.{0}.bin".format(recover_step)), map_location='cpu')
        scheduler.load_state_dict(scheduler_recover)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)

        model.train()
        if recover_step:
            start_epoch = recover_step+1
        else:
            start_epoch = 1
        for i_epoch in trange(start_epoch, int(args.num_train_epochs)+1, desc="Epoch",
                              disable=args.local_rank not in (-1, 0)):
            if args.local_rank != -1:
                train_sampler.set_epoch(i_epoch)
            iter_bar = tqdm(train_dataloader.dataset, desc='Iter (loss=X.XXX)', disable=args.local_rank not in (-1, 0))
            for step, batch in enumerate(iter_bar):
                batch = [t.to(device) if t is not None else None for t in batch]
                print(len(batch))
                input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, _ = batch
                masked_lm_loss = model(input_ids, segment_ids, input_mask, lm_label_ids,
                                       masked_pos=masked_pos, masked_weights=masked_weights)
                if n_gpu > 1:    # mean() to average on multi-gpu.
                    # loss = loss.mean()
                    masked_lm_loss = masked_lm_loss.mean()
                loss = masked_lm_loss

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

            # Save a trained model
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(
                    args.output_dir, "model.{0}.bin".format(i_epoch))
                torch.save(model_to_save.state_dict(), output_model_file)
                output_optim_file = os.path.join(
                    args.output_dir, "optim.{0}.bin".format(i_epoch))
                torch.save(optimizer.state_dict(), output_optim_file)
                if args.fp16:
                    output_amp_file = os.path.join(
                        args.output_dir, "amp.{0}.bin".format(i_epoch))
                    torch.save(amp.state_dict(), output_amp_file)
                output_sched_file = os.path.join(
                    args.output_dir, "sched.{0}.bin".format(i_epoch))
                torch.save(scheduler.state_dict(), output_sched_file)

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()


def evaluate():
    pass


