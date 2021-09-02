#coding:utf-8
import os, copy
import warnings, logging
import torch
import numpy as np
from sklearn.model_selection import KFold
from task_classes.processor import load_and_cache_examples
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models.model import save_model
from utils.util import set_seed, acc_f1_pea_spea, swa

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train(args, model, processor, tokenizer, train_dataset):
    swa_raw_model = copy.deepcopy(model)
    tb_writer = SummaryWriter()
    train_sampler = RandomSampler(train_dataset)
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
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** Running training *****")
    global_step = 0
    tr_loss, logging_loss, log_loss = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            # 减去大于1 的梯度，将其设为 1.0, 以防梯度爆炸.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            log_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                logger.info("EPOCH [%d/%d] global_step=%d loss=%f", _ + 1, args.num_train_epochs, global_step, log_loss)
                log_loss = 0.0

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # 只有在单个GPU时才评估，否则指标可能不太平均
                    if args.evaluate_during_training:
                        results = evaluate(args, processor, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, global_step)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    swa(swa_raw_model, args.output_dir, swa_start=args.swa_start)
    tb_writer.close()
    return tr_loss / global_step


def evaluate(args, model, eval_dataset, prefix=""):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    model.eval()

    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation {}*****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds, label_ids = None, None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            label_ids = np.append(label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    result = acc_f1_pea_spea(preds, label_ids)
    results.update(result)
    output_eval_file = os.path.join(eval_output_dir, "eval.txt")
    with open(output_eval_file, "w") as f:
        for key in sorted(result.keys()):
            f.write("%s = %s\n" % (key, str(result[key])))
    return results


def stack_base(args, processor, tokenizer, model, stack_train_examples, stack_dev_examples):
    train_dataset = load_and_cache_examples(args, processor, tokenizer, mode='stack', examples=stack_train_examples)
    eval_dataset = load_and_cache_examples(args, processor, tokenizer, mode='stack', examples=stack_dev_examples)
    train_loss = train(args, model, processor, tokenizer, train_dataset)
    logging.info("stack 训练结束：loss {}".format(train_loss))
    dev = evaluate(args, model, eval_dataset)
    logging.info("stack 验证结束：loss {}".format(dev))


def stacking(args, processor, tokenizer, model):
    logger.info('stacking')
    ase_output_dir = args.output_dir
    kf = KFold(5, shuffle=True, random_state=42)
    examples = processor.get_test_examples(args.data_dir)
    for i, (train_ids, dev_ids) in enumerate(kf.split(examples)):
        logger.info(f'Start to train the {i} fold')
        stack_train_examples = [examples[_idx] for _idx in train_ids]
        stack_dev_examples = [examples[_idx] for _idx in dev_ids]
        args.output_dir = os.path.join(ase_output_dir, f'stack_v{i}')
        stack_base(args, processor, tokenizer, model, stack_train_examples, stack_dev_examples)








