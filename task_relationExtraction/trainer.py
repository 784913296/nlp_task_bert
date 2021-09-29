import os
import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.util import set_seed
from models.model import save_model


logger = logging.getLogger(__name__)


def train(args, model, train_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias', 'LayerNorm.weight', 'transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    global_step = 0
    tr_loss, log_loss = 0.0, 0.0
    tb_writer = SummaryWriter()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    for epoch in train_iterator:
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'e1_mask': batch[4],
                      'e2_mask': batch[5]
                      }
            label = batch[3]
            logits = model(**inputs)
            loss = criterion(logits, label)
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
                logger.info(f"EPOCH [{epoch+1}/{args.num_train_epochs}] global_step={global_step} loss={log_loss}")
                log_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, global_step)

                if step % 10 == 9:
                    tb_writer.add_scalar('Training/training loss', tr_loss/10, epoch*len(train_dataloader)+step)
                    tr_loss = 0.0

    tb_writer.close()
    return tr_loss


def evaluate(args, model, dev_dataset, id2lable):
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
        model.eval()
        with torch.no_grad():
            tags_true = []
            tags_pred = []
            for val_i_batch, batch in enumerate(tqdm(dev_loader, desc='Validation')):
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]
                          }
                label = batch[3]
                logits = model(**inputs)
                pred_tag_ids = logits.argmax(1)
                tags_true.extend(label.tolist())
                tags_pred.extend(pred_tag_ids.tolist())

            labels = list(id2lable.keys())
            target_names = list(id2lable.values())
            result = metrics.classification_report(y_true=tags_true, y_pred=tags_pred,
                                                   labels=labels, target_names=target_names)
            output_eval_file = os.path.join(args.output_dir, args.bert_type, args.task_type, "eval.txt")
            with open(output_eval_file, "w", encoding="utf-8") as f:
                f.write(result)
            return result

