
local config = import "shared.jsonnet";

config {
    "dataset_reader"+: {
        "pretrained_tokenizer": "t5-small",
        "max_instances": 30,
    },
    "model"+: {
        "pretrained_hf_model_name": "t5-small",
    },
    "data_loader"+: {
        "batch_sampler"+: {
            "batch_size": 100
        }
    },
    "trainer"+: {
        "num_epochs": 3,
    },
}