from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/vidpr"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained vidpr downloaded from s3"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )

@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    train_dir: str = field(default=None, metadata={"help": "Path to train directory"})
    max_train_samples: int = field(
        default=None, metadata={"help": "Max number of training samples"}
    )
    valid_dir: str = field(
        default=None, metadata={"help": "Path to validation directory"}
    )
    max_valid_samples: int = field(
        default=None, metadata={"help": "Max number of validation samples"}
    )
    test_dir: str = field(
        default=None, metadata={"help": "Path to validation directory"}
    )
    max_test_samples: int = field(
        default=None, metadata={"help": "Max number of validation samples"}
    )
    passage_field_separator: str = field(default=" ")
    num_workers: int = field(
        default=4, metadata={"help": "number of process used in dataset preprocessing"}
    )
    data_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the data downloaded from huggingface"
        },
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "whether to use streaming dataset for training"},
    ) 
    object_path: str = field(
        default=None,
        metadata={"help": "objects name classes in detection"},
    )
