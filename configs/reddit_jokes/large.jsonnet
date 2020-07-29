
local config = import "shared.jsonnet";

config {
    "dataset_reader"+: {
        "pretrained_tokenizer": "t5-large",
    },
    "model"+: {
        "pretrained_hf_model_name": "t5-large",
    },
    "data_loader"+: {
        "batch_sampler"+: {
            "batch_size": 1
        }
    },
    "trainer"+: {
        "num_epochs": 5,
    },
}