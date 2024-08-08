from dataclasses import dataclass
from typing import Optional

import torch.nn.functional as F
import torch.nn as nn
import torch

from transformers import (
    PreTrainedModel, 
    RobertaModel,
    RobertaPreTrainedModel,
    DebertaV2PreTrainedModel,
    DebertaV2Model,
)

@dataclass
class SpanTokenAlignerOutput:
    """
    A data class to store the output of the BinaryTokenClassification model.

    Args:
        logits (torch.FloatTensor, *optional*): 
            The logits output by the model.
        loss (torch.FloatTensor, *optional*): 
            The computed loss, if labels are provided.
    """
    logits: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None


class AutoModelForBinaryTokenClassification:
    """
    A utility class to automatically select and load the appropriate model for binary token classification.

    Methods:
        from_pretrained(model_name_or_path: str, config): 
        Load a pretrained model given its name or path and configuration.
    """
    def from_pretrained(model_name_or_path: str, config):
        """
        Load a pretrained model given its name or path and configuration.

        Parameters:
            model_name_or_path (str): 
                The name or path of the pretrained model.
            config: 
                The configuration for the model.

        Returns:
            A model instance of either RobertaForBinaryTokenClassification or DebertaForBinaryTokenClassification.
        """
        if "roberta" in model_name_or_path:
            return RobertaForBinaryTokenClassification.from_pretrained(model_name_or_path, config=config)
        elif "deberta" in model_name_or_path:
            return DebertaForBinaryTokenClassification.from_pretrained(model_name_or_path, config=config)
        else:
            raise NotImplementedError(
                f"{model_name_or_path} not supported"
            )
        

class BinaryTokenClassification(nn.Module):
    def forward(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> SpanTokenAlignerOutput:
        
        last_hidden_state = model(input_ids, attention_mask=attention_mask)[0]
        logits = self.classifier(self.dropout(last_hidden_state)).to(torch.float32)
    
        loss = None
        if labels is not None:
            # mask then correct mean
            new_labels = torch.where(labels==-100, 0., labels)
            loss_fct = nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fct(logits.view(-1), new_labels.view(-1))
            loss = torch.where(labels.view(-1)==-100, 0, loss)
            loss = torch.sum(loss)/(labels!=-100).sum()
            
            
        return SpanTokenAlignerOutput(
            logits = logits,
            loss = loss
        )
        
        
class DebertaForBinaryTokenClassification(DebertaV2PreTrainedModel, BinaryTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.post_init()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> SpanTokenAlignerOutput:
        
        return super(DebertaForBinaryTokenClassification, self).forward(self.deberta, input_ids, attention_mask, labels)
    
        
class RobertaForBinaryTokenClassification(RobertaPreTrainedModel, BinaryTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> SpanTokenAlignerOutput:
        
        return super(RobertaForBinaryTokenClassification, self).forward(self.roberta, input_ids, attention_mask, labels)
    
