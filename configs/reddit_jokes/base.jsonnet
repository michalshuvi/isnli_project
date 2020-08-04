
local config = import "shared.jsonnet";

config {
    "dataset_reader"+: {
        "pretrained_tokenizer": "t5-base",
        "max_instances": 100,

    },
    "model"+: {
        "pretrained_hf_model_name": "t5-base",
    },
    "data_loader"+: {
        "batch_sampler"+: {
            "batch_size": 10
        }
    },
    "trainer"+: {
        "num_epochs": 5,
    },
}