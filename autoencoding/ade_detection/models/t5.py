
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

class T5(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()
        self.bert = AutoModel.from_pretrained(config.model, config=config, cache_dir='transformers_cache/')
        logger.error("\n---------The __init__ of this class will always generate warnings. "\
                     "Even if everything was initialized correctly. "\
                     "If you loaded a pretrained model using .from_pretrained() "\
                     "or loaded the state_dict, ignore the previous errors.\n---------")
    
    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None):
        if decoder_input_ids != None:
            output  = self.bert( 
                    input_ids= input_ids, 
                    attention_mask = attention_mask,
                    decoder_input_ids = decoder_input_ids, 
                    labels = labels)
            return output[0]
        else:
            output = self.bert.generate(
                    input_ids,
                    attention_mask = attention_mask,
                    max_length=32,
                    num_beams=2,
                    early_stopping=True 
                    )
            return output
