{
    "train_data_path": "datasets/reddit_jokes/reddit_jokes_train.jsonl",
    "validation_data_path": "datasets/reddit_jokes/reddit_jokes_test.jsonl",
    "dataset_reader": {
        "type": "reddit_jokes",
        "pretrained_tokenizer": error "Must override dataset_reader.pretrained_tokenizer",
    },
    "model": {
        "type": "inli",
        "pretrained_hf_model_name": error "Must override model.pretrained_hf_model"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["label_and_first_sentence"],
            "batch_size": error "Must override data_loader.batch_sampler.batch_size"
        }
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
        },
        "cuda_device": 0,
        "patience": 10,
        "checkpointer": {
            "num_serialized_models_to_keep": 3,
            "keep_serialized_model_every_num_seconds": 3600,
            "model_save_interval": 900
        }
    }
}