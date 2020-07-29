
local config = import "shared.jsonnet";

config {
    "dataset_reader"+: {
        "pretrained_tokenizer": "t5-large",
        "max_instances": 200000
    },
    "model"+: {
        "pretrained_hf_model_name": "t5-large",
    },
    "data_loader"+: {
        "batch_sampler"+: {
            "batch_size": 10
        }
    },
    "trainer"+: {
        "num_epochs": 1,
    },
}