
local config = import "shared.jsonnet";

config {
    "dataset_reader"+: {
        "pretrained_tokenizer": "t5-base",
    },
    "model"+: {
        "pretrained_hf_model_name": "t5-base",
    },
    "data_loader"+: {
        "batch_sampler"+: {
            "batch_size": 30
        }
    },
    "trainer"+: {
        "num_epochs": 2,
    },
}