from transformers import BertPreTrainedModel, BertForTokenClassification, AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F
from torchcrf import CRF
from torch import nn
import torch
log_soft = F.log_softmax

import logging
logger = logging.getLogger(__name__)
import transformers
transformers.logging.set_verbosity(transformers.logging.CRITICAL)

class Bert_LSTM(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.dropout)
        self.bert_out_to_labels = nn.Linear(config.hidden_size*2, self.num_labels)
        self.config_size = config.hidden_size
        # ultimo embedding di LSTM
        self.final_lstm = nn.LSTM(config.hidden_size,
                                  config.hidden_size,
                                  num_layers=1,
                                  bidirectional=True,
                                  dropout=config.dropout,
                                  batch_first=True)
        self.init_weights()
        self.bert = AutoModel.from_pretrained(config.model, config=config)

    def _get_lstm_out(self, emb_input, attention_mask):
        mask = attention_mask.detach().cpu().numpy().tolist()
        mask = [elem.index(0) if 0 in elem else len(elem) for elem in mask]
        packed_x = pack_padded_sequence(emb_input, mask, batch_first=True, enforce_sorted=False)
        lstm_out,_ = self.final_lstm(packed_x)
        padded_x,_ = pad_packed_sequence(lstm_out, batch_first=True, total_length=emb_input.shape[1])
        return padded_x
  

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        bert_output = outputs[0]
        lstm_output = self._get_lstm_out(bert_output, attention_mask) 
        probs = self.bert_out_to_labels(self.dropout(lstm_output))

        logits = log_soft(probs, 2)
        output = None
        
        preds = logits.argmax(axis=-1)
        
        outputs = (preds,)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, logits.shape[2]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)
    
