import logging
import os
import numpy as np
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from task_ner.processor import real_labels
from utils.util import set_seed, flatten
from models.model import save_model
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train(args, train_dataset, eval_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    model.train()

    no_decay = ['bias', 'LayerNorm.weight', 'transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    logger.info("***** 训练中 *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    best_f1 = 0.
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            loss, pre_tag = outputs[0], outputs[1]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            logging_loss += loss.item()
            tr_loss += loss.item()
            if 0 == (step + 1) % args.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                logger.info("EPOCH [%d/%d] global_step=%d   loss=%f", _+1, args.num_train_epochs, global_step, logging_loss)
                logging_loss = 0.0

                # 每相隔 args.save_steps 步，评估一次
                if global_step % args.save_steps == 0:
                    # best_f1 = evaluate_and_save_model(args, model, eval_dataset, _, global_step, best_f1)
                    save_model(args, model, global_step)


    # 最后循环结束, 评估一次
    best_f1 = evaluate_and_save_model(args, model, eval_dataset, _, global_step, best_f1)
    return best_f1


def evaluate_and_save_model(args, model, eval_dataset, epoch, global_step, best_f1):
    ret = evaluate(args, model, eval_dataset)

    precision_b = ret['1']['precision']
    recall_b = ret['1']['recall']
    f1_b = ret['1']['f1-score']
    support_b = ret['1']['support']

    precision_i = ret['2']['precision']
    recall_i = ret['2']['recall']
    f1_i = ret['2']['f1-score']
    support_i = ret['2']['support']

    weight_b = support_b / (support_b + support_i)
    weight_i = 1 - weight_b

    avg_precision = precision_b * weight_b + precision_i * weight_i
    avg_recall = recall_b * weight_b + recall_i * weight_i
    avg_f1 = f1_b * weight_b + f1_i * weight_i
    all_avg_precision = ret['macro avg']['precision']
    all_avg_recall = ret['macro avg']['recall']
    all_avg_f1 = ret['macro avg']['f1-score']

    logger.info("Evaluating EPOCH = [%d/%d] global_step = %d", epoch+1,args.num_train_epochs, global_step)
    logger.info("B-LOC precision = %f recall = %f  f1 = %f support = %d", precision_b, recall_b, f1_b, support_b)
    logger.info("I-LOC precision = %f recall = %f  f1 = %f support = %d", precision_i, recall_i, f1_i, support_i)
    logger.info("attention AVG:precision = %f recall = %f  f1 = %f ", avg_precision, avg_recall, avg_f1)
    logger.info("all AVG:precision = %f recall = %f  f1 = %f ", all_avg_precision, all_avg_recall, all_avg_f1)

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
        logging.info("save the best model %s,avg_f1= %f", os.path.join(args.output_dir, "pytorch_model.bin"), best_f1)

    return best_f1


def evaluate(args, model, eval_dataset):
    eval_output_dirs = args.output_dir
    if not os.path.exists(eval_output_dirs):
        os.makedirs(eval_output_dirs)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()

    logger.info("***** 验证中 *****")
    loss = []
    y_true_list = []
    y_pred_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]
                      }
            outputs = model(**inputs)
        eval_loss, logits = outputs[:2]
        list_tmp = []
        list_tmp.append(eval_loss.tolist())
        loss.extend(list_tmp)

        logits = model.crf.decode(logits, inputs['attention_mask'])
        labels_pred = logits.squeeze(0).cpu().tolist()
        labels_true = batch[3].tolist()

        labels_pred_real = list(real_labels(inputs['attention_mask'], labels_pred))
        labels_true_real = list(real_labels(inputs['attention_mask'], labels_true))
        y_pred_list.extend(labels_pred_real)
        y_true_list.extend(labels_true_real)

    y_true = np.array(flatten(y_true_list))
    y_pred = np.array(flatten(y_pred_list))
    assert y_true.shape == y_pred.shape, "y_true vs y_pred {} vs {}".format(len(y_true), len(y_pred))
    ret = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    return ret


'''
#               precision    recall  f1-score   support
#
#            0   0.998345  0.996229  0.997286     89638
#            1   0.993100  0.989685  0.991389      9016
#            2   0.992506  0.997225  0.994860     46483
#
#    micro avg   0.996142  0.996142  0.996142    145137
#    macro avg   0.994650  0.994380  0.994512    145137
# weighted avg   0.996149  0.996142  0.996143    145137
'''
def test(args, test_dataset, model):
    sampler = RandomSampler(test_dataset)
    data_loader = DataLoader(test_dataset, sampler=sampler, batch_size=2)
    y_true_list = []
    y_pred_list = []

    for batch in tqdm(data_loader, desc="test"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': None
                      }
            outputs = model(**inputs)
        logits = outputs[0]
        logits = model.crf.decode(logits, inputs['attention_mask'])
        labels_pred = logits.squeeze(0).cpu().tolist()
        labels_true = batch[3].tolist()

        labels_pred_real = list(real_labels(inputs['attention_mask'], labels_pred))
        labels_true_real = list(real_labels(inputs['attention_mask'], labels_true))
        y_pred_list.extend(labels_pred_real)
        y_true_list.extend(labels_true_real)

    y_pred = np.array(flatten(y_pred_list))
    y_true = np.array(flatten(y_true_list))
    assert y_true.shape == y_pred.shape
    ret = classification_report(y_true=y_true, y_pred=y_pred, digits=6)
    return ret