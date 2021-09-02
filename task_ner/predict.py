import logging
import torch
from task_ner.processor import NerProcessor
from utils.ner_utils import label2id_load, get_entities, get_entitys_json
from task_ner.conf import args_ner
from models.model import create_model, create_tokenizer
import warnings


def predict_ner(args, model, tokenizer, sentence, max_len=64):
    sentence_list = list(sentence.strip().replace(' ', ''))
    text = " ".join(sentence_list)
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_len,
                                   truncation=True)

    input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(args.device)
    attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(args.device)
    token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(args.device)
    model = model.to(args.device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs[0]
        tags = model.crf.decode(logits, attention_mask)
        tags = tags.squeeze(0).cpu().tolist()[0]

    labelsid_list = tags[1:-1]  # [CLS]XXX[SEP]
    assert len(labelsid_list) == len(sentence_list) or len(labelsid_list) == max_len - 2
    labels_list = [args.id2label[labelid] for labelid in labelsid_list]
    entity = get_entities(sentence, labels_list, markup='bioes')
    return entity


def predict_ner2json(args, model, tokenizer, sentence, max_len=64):
    sentence_list = list(sentence.strip().replace(' ', ''))
    text = " ".join(sentence_list)
    inputs = tokenizer.encode_plus(text, add_special_tokens=True, max_length=max_len,
                                   truncation=True)

    input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs["attention_mask"]
    input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(args.device)
    attention_mask = torch.LongTensor(attention_mask).unsqueeze(0).to(args.device)
    token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(args.device)
    model = model.to(args.device)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs[0]
        tags = model.crf.decode(logits, attention_mask)
        tags = tags.squeeze(0).cpu().tolist()[0]

    labelsid_list = tags[1:-1]  # [CLS]XXX[SEP]
    assert len(labelsid_list) == len(sentence_list) or len(labelsid_list) == max_len - 2
    labels_list = [args.id2label[labelid] for labelid in labelsid_list]
    entity = get_entitys_json(sentence, labels_list, markup='bioes')
    return entity



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logger = logging.getLogger(__name__)
    args = args_ner()
    args.label2id = label2id_load()
    args.id2label = {v: k for k, v in args.label2id.items()}
    args.num_labels = len(args.label2id)

    processor = NerProcessor()
    tokenizer = create_tokenizer(args)
    model_file = "../baseline/albert/task_ner/pytorch_model.bin"
    model = create_model(args, model_file)
    model.to(args.device)

    if args.do_pred:
        # test_dataset = load_and_cache_example(args, tokenizer, processor, 'test')
        # print(test(args, test_dataset, model))
        # sentence = "做辣子鸡需要准备什么食材"
        sentence = "宝宝随申码注册不了"  # 随申码 注册
        entity = predict_ner2json(args, model, tokenizer, sentence)
        print(entity)