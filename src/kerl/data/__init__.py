from .redial import WikiRedial
from .inspired import WikiInspired
from .wikimkg import WikiMKGDataset, WikiMKGDataModule

DATASET_REGISTRY = {
    "redial": WikiRedial,
    "inspired": WikiInspired,
}


def get_dataset(dataset_name: str):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} is not registered. Only {list(DATASET_REGISTRY.keys())} are valid.")
    return DATASET_REGISTRY[dataset_name]
