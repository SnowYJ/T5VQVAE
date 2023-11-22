from dataclasses import dataclass, field
from transformers import TrainingArguments, MODEL_WITH_LM_HEAD_MAPPING
from typing import Optional


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class MyTrainingArguments(TrainingArguments):
    project_name: str = field(default=None, metadata={"help": "The Weights & Biases project name for the run."})
    reg_schedule_k: float = field(default=0.0025, metadata={"help": "Multiplied by global_step in a sigmoid, more gradually increase regulariser loss weight."})
    reg_schedule_b: float = field(default=6.25, metadata={"help": "Added to global step in sigmoid, further delays increase in regulariser loss weight."})
    reg_constant_weight: Optional[float] = field(default=None, metadata={"help": "Apply a constant weight to the regulariser."})
    use_recon_loss: bool = field(default=False, metadata={"help": "Have the reconstructed encodings match their input encodings."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    t5_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the T5 model being using for encoding & decoding."
        },
    )

    model_type: Optional[str] = field(
        default='t5',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    ae_latent_size: int = field(default=None, metadata={"help": "The size of the VAE's latent space, only valid with a T5 model."})
    set_seq_size: int = field(default=None, metadata={"help": "Set sequence size, needed for VAE compression."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


if __name__ == '__main__':
    model_args = ModelArguments(model_path='', t5_model_name='t5-base', ae_latent_size=1000, set_seq_size=70)
    data_args = DataTrainingArguments(train_data_file='tr-data.csv')
    training_args = MyTrainingArguments(project_name="T5-VAE",
                                        output_dir='output',
                                        do_train=True,
                                        per_device_train_batch_size=28,
                                        gradient_accumulation_steps=7,
                                        save_total_limit=1,
                                        save_steps=-1,
                                        num_train_epochs=3,
                                        logging_steps=-1,
                                        overwrite_output_dir=True,)

    print(model_args.__dict__)
    print(data_args.__dict__)
    print(training_args.__dict__)
