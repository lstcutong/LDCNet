import torch
import torch.nn as nn
from typing import List, Type
from ..builder import MODELS
from pointpd.models.losses import build_criteria
import copy

@MODELS.register_module()
class BaseSemanticSegmentor(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 head,
                 criteria
                 ):
        super().__init__()

        self.encoder = MODELS.build(encoder)
        decoder_args_merged_with_encoder = copy.deepcopy(encoder)
        decoder_args_merged_with_encoder.update(decoder)
        self.decoder = MODELS.build(decoder_args_merged_with_encoder)
        self.head = MODELS.build(head)
        self.criteria = build_criteria(criteria)

    
    def forward(self, input_dict):
        logits = self.head(self.decoder(self.encoder(input_dict)))


        if "segment" in input_dict.keys():
            loss = self.criteria(logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=logits)
        else:
            return dict(seg_logits=logits)


