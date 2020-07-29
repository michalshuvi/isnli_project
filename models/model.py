from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data.fields import MetadataField

from allennlp.models import Model
from transformers import T5ForConditionalGeneration


@Model.register('isnli')
class IsnliModel(Model):
    def __init__(self, vocab: Vocabulary,
                 pretrained_hf_model_name: str):
        super().__init__(vocab)
        self.pretrained_hf_model = T5ForConditionalGeneration.from_pretrained(pretrained_hf_model_name,
                                                                              cache_dir='/home/ML_courses/course_DBNLP/michalshuvi')

    def forward(self, class_and_premise: Dict[str, Dict[str, torch.Tensor]],
                hypothesis: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        output_dict = {}
        class_and_premise_ids = class_and_premise['tokens']['token_ids']
        hypothesis_ids = hypothesis['tokens']['token_ids']

        if self.training:
            output = self.pretrained_hf_model(input_ids=class_and_premise_ids, lm_labels=hypothesis_ids)
            output_dict['loss'] = output[0]

        else:
            with torch.no_grad():
                predicted_ids = self.pretrained_hf_model.generate(class_and_premise_ids)
                print(predicted_ids)
                output = self.pretrained_hf_model(input_ids=class_and_premise_ids, lm_labels=hypothesis_ids)
                output_dict['loss'] = output['loss']

        return output_dict

    def generate(self, input_ids):
        return self.pretrained_hf_model.generate(input_ids)
