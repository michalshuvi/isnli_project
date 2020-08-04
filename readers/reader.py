import json
import pickle

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from typing import Iterator, Iterable

from allennlp.data.fields import TextField
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from overrides import overrides


class InliDatasetReaderBase(DatasetReader):
    def __init__(self, pretrained_tokenizer: str, max_instances: int = None):
        super().__init__(max_instances=max_instances)
        self.tokenizer = PretrainedTransformerTokenizer(pretrained_tokenizer, max_length=2000)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(pretrained_tokenizer)}

    def text_to_instance(self, data_chunk) -> Instance:
        raise NotImplementedError

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, encoding="utf8") as jsonl_file:
            dataset_list = list(jsonl_file)

        for data_chunk in dataset_list:
            yield self.text_to_instance(json.loads(data_chunk))

    def prepare_fields(self, task_name: str, label: str, sentence1: str, sentence2: str):
        label_and_first_sentence_tokens = self.tokenizer.tokenize(f'{task_name} {label}: {sentence1} '
                                                                  f'{self.tokenizer.tokenizer.eos_token}')
        second_sentence_tokens = self.tokenizer.tokenize(f'{sentence2} {self.tokenizer.tokenizer.eos_token}')
        fields = {
            "label_and_first_sentence": TextField(label_and_first_sentence_tokens, self.token_indexers),
            "second_sentence": TextField(second_sentence_tokens, self.token_indexers),
        }
        return fields


@DatasetReader.register('snli')
class SnliDatasetReader(InliDatasetReaderBase):

    def text_to_instance(self, data_chunk) -> Instance:
        premise = data_chunk['sentence1']
        hypothesis = data_chunk['sentence2']
        label = data_chunk['gold_label']
        fields = self.prepare_fields('inli', label, premise, hypothesis)
        return Instance(fields)


@DatasetReader.register('boolq')
class BoolQDatasetReader(InliDatasetReaderBase):

    def text_to_instance(self, data_chunk) -> Instance:
        answer = data_chunk['answer']
        question = data_chunk['question']
        passage = data_chunk['passage']
        fields = self.prepare_fields('generate question', answer, passage, question)
        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename: str) -> Iterable[Instance]:
        print('reading instances from', cache_filename)
        with open(cache_filename, 'rb') as cache_file:
            instances = pickle.load(cache_file)
            for instance in instances:
                yield instance

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances) -> None:
        print('writing instance to', cache_filename)
        with open(cache_filename, 'wb') as cache_file:
            pickle.dump(instances, cache_file)


@DatasetReader.register('reddit_jokes')
class RedditJokesDatasetReader(InliDatasetReaderBase):

    def text_to_instance(self, data_chunk) -> Instance:
        post_title_and_text = data_chunk['post_title'] + ". " + data_chunk['post_text']
        comment = data_chunk['comment']
        label = data_chunk['label']
        fields = self.prepare_fields('comment', label, post_title_and_text, comment)
        return Instance(fields)
