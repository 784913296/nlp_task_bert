import re
import torch
from task_relationExtraction.conf import args_relation_extraction
from task_relationExtraction.util import get_id2label
from task_relationExtraction.processor import get_pos, convert_pos_to_mask
from models.model import create_model, create_tokenizer


def predict(args, model, tokenizer, text, entity1, entity2):
    model.eval()
    match_obj1 = re.search(entity1, text)
    match_obj2 = re.search(entity2, text)

    # 使用第一个匹配的实体的位置
    if match_obj1 and match_obj2:
        e1_pos = match_obj1.span()
        e2_pos = match_obj2.span()
        context = {
            'h': {'name': entity1, 'pos': e1_pos},
            't': {'name': entity2, 'pos': e2_pos},
            'text': text
        }

        token, pos_e1, pos_e2 = get_pos(context, tokenizer)
        encoded = tokenizer.batch_encode_plus([(token, None)], return_tensors='pt')
        input_ids = encoded['input_ids'].to(args.device)
        token_type_ids = encoded['token_type_ids'].to(args.device)
        attention_mask = encoded['attention_mask'].to(args.device)
        e1_mask = torch.tensor([convert_pos_to_mask(pos_e1, max_length=attention_mask.shape[1])]).to(args.device)
        e2_mask = torch.tensor([convert_pos_to_mask(pos_e2, max_length=attention_mask.shape[1])]).to(args.device)

        with torch.no_grad():
            logits = model(input_ids, token_type_ids, attention_mask, e1_mask, e2_mask)[0]
            logits = logits.to(torch.device('cpu'))
        pred = args.id2label[logits.argmax(0).item()]
        print(f"最大可能的关系是：{pred}")
        top_ids = logits.argsort(0, descending=True).tolist()
        for i, tag_id in enumerate(top_ids, start=1):
            print(f"No.{i}: {args.id2label[tag_id]} 的可能性：{logits[tag_id]}")
    else:
        if match_obj1 is None:
            print('实体1不在句子中')
        if match_obj2 is None:
            print('实体2不在句子中')


if __name__ == '__main__':
    args = args_relation_extraction()
    args.id2label = get_id2label(args.labels_file)
    args.num_labels = len(args.id2label)
    tokenizer = create_tokenizer(args)
    model_file = "../model_data/albert/task_relationExtraction/checkpoint-1400/pytorch_model.bin"
    model = create_model(args, model_file)
    text = "这些年从运动员到普通人的转变如此成功，于芬一边劝慰伏明霞，说没有当年的苛刻也不可能成就自己，一边笑着邀请于芬到香港玩"
    entity1 = "于芬"
    entity2 = "伏明霞"
    predict(args, model, tokenizer, text, entity1, entity2)
