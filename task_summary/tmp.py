import torch
from transformers import BertTokenizer
import collections


max_length = 128
def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    """ 截断超过 max_len-3 的句子 """
    if len(tokens_a) + len(tokens_b) > max_len-3:
        while len(tokens_a) + len(tokens_b) > max_len-3:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
    return tokens_a, tokens_b

tokens_a = "日前，国家核安全局连发两项通知，上海和扬州的两家核设备公司审批许可被否。据了解，这是近3年来，国家核安全局首次在核安全设备问题上直接说“不”。专家预计未来对民用核安全的监管将会加强。"
tokens_b = "民用核安全管理3年来首次趋紧"

# tokens_a = '目前中国大陆境内18洞标准高尔夫球场的数量已增至490个，其中只有10个得到政府立项批准---中国高尔夫行业组织朝向管理集团周一发表报告称。国土资源部2004年就曾明令禁止修建高尔夫球场，国土资源部国家土地副总督察甘藏春上个月表示这个禁令目前依然有效。'
# tokens_b = '18亿亩红线限不住高尔夫球场'

tokens_a, tokens_b = truncate_tokens_pair(tokens_a, tokens_b, max_length)

tokenizer = BertTokenizer.from_pretrained("../baseline/unilm_chinese")
tokenizer_inputs = tokenizer.encode_plus(tokens_a, tokens_b, add_special_tokens=True, max_length=max_length,
                                         pad_to_max_length=True, truncation="longest_first")
token_type_ids = tokenizer_inputs["token_type_ids"]
print(token_type_ids)
print(collections.Counter(token_type_ids))


# _tril_matrix = torch.tril(torch.ones((max_length, max_length), dtype=torch.long))
# attention_mask = torch.zeros(max_length, max_length, dtype=torch.long)
# attention_mask[:, :len(tokens_a) + 2].fill_(1)
# second_st, second_end = len(tokens_a) + 2, len(tokens_a) + len(tokens_b) + 3
# attention_mask[second_st:second_end, second_st:second_end].copy_(
#     _tril_matrix[:second_end - second_st, :second_end - second_st])
# print(attention_mask[:1])
# print(attention_mask[-15:-10])

