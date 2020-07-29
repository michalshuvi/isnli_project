import json
import pickle

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from typing import Iterator, Iterable

from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from overrides import overrides


@DatasetReader.register('isnli')
class SnliDatasetReader(DatasetReader):
    def __init__(self, pretrained_tokenizer: str, max_instances: int = None):
        super().__init__(max_instances=max_instances)
        self.tokenizer = PretrainedTransformerTokenizer(pretrained_tokenizer)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(pretrained_tokenizer)}

    def text_to_instance(self, data_chunk) -> Instance:
        premise = data_chunk['sentence1']
        hypothesis = data_chunk['sentence2']
        label = data_chunk['gold_label']
        class_and_premise_tokens = self.tokenizer.tokenize(f'isnli {label}: {premise} '
                                                           f'{self.tokenizer.tokenizer.eos_token}')
        hypothesis_tokens = self.tokenizer.tokenize(f'{hypothesis} {self.tokenizer.tokenizer.eos_token}')
        fields = {
            "class_and_premise": TextField(class_and_premise_tokens, self.token_indexers),
            "hypothesis": TextField(hypothesis_tokens, self.token_indexers),
        }
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path) as jsonl_file:
            dataset_list = list(jsonl_file)

        for data_chunk in dataset_list:
            yield self.text_to_instance(json.loads(data_chunk))


@DatasetReader.register('boolq')
class BoolQDatasetReader(DatasetReader):
    def __init__(self, pretrained_tokenizer: str, max_instances: int = None, cache_directory: str = 'data/cache'):
        super().__init__(max_instances=max_instances, cache_directory=cache_directory)
        self.tokenizer = PretrainedTransformerTokenizer(pretrained_tokenizer, max_length=2000)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(pretrained_tokenizer)}

    def text_to_instance(self, data_chunk) -> Instance:
        answer = data_chunk['answer']
        question = data_chunk['question']
        passage = data_chunk['passage']
        answer_and_passage_tokens = self.tokenizer.tokenize(f'generate question {answer}: {passage} '
                                                            f'{self.tokenizer.tokenizer.eos_token}')
        question_tokens = self.tokenizer.tokenize(f'{question} {self.tokenizer.tokenizer.eos_token}')
        fields = {
            "class_and_premise": TextField(answer_and_passage_tokens, self.token_indexers),
            "hypothesis": TextField(question_tokens, self.token_indexers),
        }
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path, encoding="utf8") as jsonl_file:
            dataset_list = list(jsonl_file)

        for data_chunk in dataset_list:
            yield self.text_to_instance(json.loads(data_chunk))


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
class RedditJokesDatasetReader(DatasetReader):
    def __init__(self, pretrained_tokenizer: str, max_instances: int = None):
        super().__init__(max_instances=max_instances)
        self.tokenizer = PretrainedTransformerTokenizer(pretrained_tokenizer, max_length=512)
        self.token_indexers = {"tokens": PretrainedTransformerIndexer(pretrained_tokenizer)}

    def text_to_instance(self, data_chunk) -> Instance:
        post_title_and_text = data_chunk['post_title'] + ". "+ data_chunk['post_text']
        comment = data_chunk['comment']
        label = data_chunk['label']
        post_and_label_tokens = self.tokenizer.tokenize(f'reddit jokes {label}: {post_title_and_text} '
                                                           f'{self.tokenizer.tokenizer.eos_token}')
        comment_tokens = self.tokenizer.tokenize(f'{comment} {self.tokenizer.tokenizer.eos_token}')
        fields = {
            "class_and_premise": TextField(post_and_label_tokens, self.token_indexers),
            "hypothesis": TextField(comment_tokens, self.token_indexers),
        }
        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path) as jsonl_file:
            dataset_list = list(jsonl_file)

        for data_chunk in dataset_list:
            yield self.text_to_instance(json.loads(data_chunk))
