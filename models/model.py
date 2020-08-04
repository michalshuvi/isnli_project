from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data.fields import MetadataField

from allennlp.models import Model
from transformers import T5ForConditionalGeneration


@Model.register('inli')
class InliModel(Model):
    def __init__(self, vocab: Vocabulary,
                 pretrained_hf_model_name: str,
                 cache_dir=None):
        super().__init__(vocab)
        self.pretrained_hf_model = T5ForConditionalGeneration.from_pretrained(pretrained_hf_model_name,
                                                                              cache_dir=cache_dir)

    def forward(self, label_and_first_sentence: Dict[str, Dict[str, torch.Tensor]],
                second_sentence: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        output_dict = {}
        label_and_first_sentence_ids = label_and_first_sentence['tokens']['token_ids']
        second_sentence_ids = second_sentence['tokens']['token_ids']

        if self.training:
            output = self.pretrained_hf_model(input_ids=label_and_first_sentence_ids, lm_labels=second_sentence_ids)
            output_dict['loss'] = output[0]

        else:
            with torch.no_grad():
                output = self.pretrained_hf_model(input_ids=label_and_first_sentence_ids, lm_labels=second_sentence_ids)
                output_dict['loss'] = output[0]

        return output_dict

    def generate(self, input_ids, **args):
        return self.pretrained_hf_model.generate(input_ids, **args)
