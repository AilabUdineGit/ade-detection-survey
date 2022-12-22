from transformers import BertPreTrainedModel, BertForSequenceClassification, AutoModel
import torch.nn.functional as F
from torchcrf import CRF
from torch import nn
import torch
log_soft = F.log_softmax

import logging
logger = logging.getLogger(__name__)

class Bert_wrapper_binary(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        config.num_classes = 2
        
        self.init_weights()
        self.bert = BertForSequenceClassification.from_pretrained(config.model, config=config)
        logger.error("\n---------The __init__ of this class will always generate warnings. "\
                     "Even if everything was initialized correctly. "\
                     "If you loaded a pretrained model using .from_pretrained() "\
                     "or loaded the state_dict, ignore the previous errors.\n---------")
    
    def forward(self, input_ids, attention_mask, labels=None):
        
        out = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
        )

        if labels != None:
        
            loss, logits = out
            preds = logits.argmax(axis=-1)
            return (loss, preds)
        
        else:
            
            logits = out[0]
            preds = logits.argmax(axis=-1)
            return preds