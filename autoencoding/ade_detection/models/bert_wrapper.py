from transformers import BertPreTrainedModel, BertForTokenClassification, AutoModel, AutoModelForTokenClassification
import torch.nn.functional as F
from torchcrf import CRF
from torch import nn
import torch
log_soft = F.log_softmax
import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/HDD/transformers/'

import logging
logger = logging.getLogger(__name__)

import transformers
transformers.logging.set_verbosity(transformers.logging.CRITICAL)

class Bert_wrapper(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()
        self.bert = AutoModelForTokenClassification.from_pretrained(config.model, config=config, cache_dir='transformers_cache/')
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
        
            loss, logits = out[0], out[1]
            preds = logits.argmax(axis=-1)
            return (loss, preds)
        
        else:
            
            logits = out[0]
            preds = logits.argmax(axis=-1)
            return preds