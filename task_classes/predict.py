# coding:utf-8
import os
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SequentialSampler
from task_classes.conf import Args
from task_classes.processor import classesProcessor, load_and_cache_examples
from utils.util import filterContent
from models.model import create_model, EnsembleModel, last_model_file, create_tokenizer

args = Args().get_parser()

def text2tokenizer(text, tokenizer):
    text = str(text).strip()
    sentencses = ILLEGAL_CHARACTERS_RE.sub(r'', text)
    sequence_dict = tokenizer.encode_plus(sentencses, max_length=args.max_length, pad_to_max_length=True,
                                          padding=True, truncation=True)
    input_ids = sequence_dict['input_ids']
    attention_mask = sequence_dict['attention_mask']
    token_type_ids = sequence_dict["token_type_ids"]

    token_ids = torch.LongTensor(input_ids).unsqueeze(0)
    token_mask = torch.LongTensor(attention_mask).unsqueeze(0)
    token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0)

    inputs = {
        'input_ids': token_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': token_mask
    }
    return inputs



def one_text_predict(text, model, tokenizer, index2cid):
    """ 输入的 text 为单条形式 """
    model.eval()
    inputs = text2tokenizer(text, tokenizer)
    with torch.no_grad():
        logits = model(**inputs)

    _, predict = logits[0].max(1)
    label = index2cid[predict.item()]
    return label


def text_predict(examples, model, tokenizer, index2cid):
    """ 输入的 text 为列表形式 """
    model.eval()
    labels_list = []
    inputs_list = []

    for text in examples:
        inputs = text2tokenizer(text, tokenizer)
        inputs_list.append(inputs)

    for inputs in inputs_list:
        with torch.no_grad():
            logits = model(**inputs)
        _, predict = logits[0].max(1)
        label = index2cid[predict.item()]
        labels_list.append(label)
    return labels_list



def base_predict(test_dataset, model, index2cid, ensemble=False, vote=False):
    sampler = SequentialSampler(test_dataset)
    eval_dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=1)

    labels_list = []
    for batch in tqdm(eval_dataloader, desc="Predicting"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }

            if ensemble:
                if vote:
                    predict = model.vote_predict(inputs=inputs)
                    label = index2cid[predict]
                else:
                    predict = model.predict(inputs=inputs)
                    label = index2cid[predict.item()]
            else:
                logits = model(**inputs)
                _, predict = logits[0].max(1)
                label = index2cid[predict.item()]

        labels_list.append(label)
    return labels_list


def single_predict(test_dataset, model, index2cid):
    model.eval()
    labels = base_predict(test_dataset, model, index2cid)
    return labels



def ensemble_predict(test_dataset, model, id2label, vote=True):
    """ 投票/加权 """

    # ensemble_model_list.txt 模型路径列表
    with open('./ensemble_model_list.txt', 'r', encoding='utf-8') as f:
        ensemble_dir_list = f.readlines()
        print('ENSEMBLE_DIR_LIST:{}'.format(ensemble_dir_list))
    model_path_list = [x.strip() for x in ensemble_dir_list]
    print('model_path_list:{}'.format(model_path_list))

    model = EnsembleModel(model=model, model_path_list=model_path_list, device=args.device, lamb=lamb)
    labels = base_predict(test_dataset, model, id2label, ensemble=True, vote=True)
    return labels


if __name__ == '__main__':
    lamb = 0.3
    threshold = 0.9
    filter_content = filterContent()
    processor = classesProcessor()
    index2cid = processor.get_index2cid()

    output_dir = os.path.join(args.output_dir, args.bert_type, args.task_type)
    last_model_top5 = last_model_file(output_dir, count=5)
    with open('./ensemble_model_list.txt', 'w', encoding='utf-8') as f:
        for _file in last_model_top5:
            f.write(_file + "\n")

    model_file = last_model_file(output_dir, count=1)[0]
    model = create_model(args, model_file)
    tokenizer = create_tokenizer(args)
    test_dataset = load_and_cache_examples(args, processor, tokenizer, mode='test')

    labels_list = single_predict(test_dataset, model, index2cid)
    print(labels_list)

    labels_list = ensemble_predict(test_dataset, model, index2cid)
    print(labels_list)

    text = "我这里网不好地址写错了"
    label = one_text_predict(text, model, tokenizer, index2cid)
    print(label)

    text = ["我这里网不好地址写错了", "亲我收到的东西都摔烂了", "我想问一下删除订单的话怎么删除", "我地址忘记改了能不能换地址"]
    label_list = text_predict(text, model, tokenizer, index2cid)
    print(label_list)