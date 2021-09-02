from transformers import AlbertModel, AlbertPreTrainedModel
from crf import CRF
import torch.nn as nn


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