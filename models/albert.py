import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertPreTrainedModel
from task_ner.crf import CRF


class AlbertCrfForNer(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertCrfForNer, self).__init__(config)
        self.bert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)

        # 这里在seq_len的维度上去头，是去掉了[CLS]，去尾巴有两种情况: 1、是 <pad> 2、[SEP]
        new_sequence_output = sequence_output[:, 1:-1]
        new_attention_mask = attention_mask[:, 2:].bool()

        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores


class AlbertSentenceReModel(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        hidden_size = config.hidden_size
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(hidden_size * 3)
        self.hidden2tag = nn.Linear(hidden_size * 3, config.num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        sequence_output, pooled_output = self.albert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                     attention_mask=attention_mask, return_dict=False)

        # 每个实体的所有token向量的平均值
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        e1_h = self.activation(self.dense(e1_h))
        e2_h = self.activation(self.dense(e2_h))

        # [cls] + 实体1 + 实体2
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        concat_h = self.norm(concat_h)
        logits = self.hidden2tag(self.dropout(concat_h))

        return logits

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        # [b, 1, j-i+1]
        e_mask_unsqueeze = e_mask.unsqueeze(1)
        # [batch_size, 1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector