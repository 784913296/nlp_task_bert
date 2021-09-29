# coding=utf-8
from __future__ import absolute_import, division, print_function
import os
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from task_mrc.util import EvalOps, RawResult
from task_mrc.write_predict import write_predictions
from task_mrc.utils_squad_eval import main as evaluate_on_squad
from models.model import save_model
from utils.util import set_seed

logger = logging.getLogger(__name__)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 多gpu训练(应该在apex fp16初始化之后)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 分布式训练(应该在apex fp16初始化之后)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps *
                (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss, log_loss = 0.0, 0.0, 0.0
    model.zero_grad()
    local_rank = args.local_rank in [-1, 0]
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'start_positions': batch[3],
                      'end_positions': batch[4]}
            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                # mean()对多gpu并行(非分布式)训练的平均值
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            log_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                logger.info("EPOCH [%d/%d] global_step=%d loss=%f", _ + 1,
                            args.num_train_epochs, global_step, log_loss)
                log_loss = 0.0

                if local_rank and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 只在单个GPU时进行评估，否则指标可能不平均
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if local_rank and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if local_rank:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, dataset, examples, features, prefix=""):
    output_dir = os.path.join(args.output_dir, args.bert_type, args.task_type)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            start_logits = to_list(outputs[0][i])
            end_logits = to_list(outputs[1][i])
            result = RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits)
            all_results.append(result)

    # Compute predictions
    output_prediction_file = os.path.join(output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(output_dir, "nbest_predictions_{}.json".format(prefix))
    output_null_log_odds_file = None
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(output_dir, "null_odds_{}.json".format(prefix))

    write_predictions(examples, features, all_results, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args)

    data_file = os.path.join(args.data_dir, "cmrc2018_dev.json")
    evaluate_options = EvalOps(data_file=data_file, pred_file=output_prediction_file,
                               na_prob_file=output_null_log_odds_file)
    results = evaluate_on_squad(evaluate_options)
    return results
