import re


class InputExample:
    def __init__(self, guid, text, label):
        self.guid = guid
        self.label = label
        self.text = text


class InputFeatures:
    def __init__(self, input_ids, attention_mask, token_type_ids, label, e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask


def get_id2label(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def get_label2id(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))