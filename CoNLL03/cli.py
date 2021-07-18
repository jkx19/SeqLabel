import datasets
from datasets.load import load_metric, load_dataset, load_dataset_builder
import numpy as np
import torch
from torch import Tensor
import torch.nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.optim import AdamW, lr_scheduler
from transformers import TrainingArguments, HfArgumentParser
from transformers.trainer_pt_utils import get_parameter_names

from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional
import os

from data.conll_dataset import CoNLL
from model.prefix import BertForTokenClassification
from trainer import Trainer


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='bert-base-uncased',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    # def __post_init__(self):
    #     if self.dataset_name is None and self.train_file is None and self.validation_file is None:
    #         raise ValueError("Need either a dataset name or a training/validation file.")
    #     else:
    #         if self.train_file is not None:
    #             extension = self.train_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
    #         if self.validation_file is not None:
    #             extension = self.validation_file.split(".")[-1]
    #             assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    #     self.task_name = self.task_name.lower()


# METRIC: F1 score
# Note: the main reason abandoning LAMA is to fit the metric

class Trainer_API:    
    def __init__(self) -> None:

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))        
        self.model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses()

        self.task = 'ner'
        assert self.task in ['pos', 'chunk', 'ner']
        self.device = torch.device('cuda:0')

        self.batch_size = 4
        self.epoch = 10
        self.adam_beta1 = 0.01
        self.adam_beta2 = 0.01
        self.adam_epsilon = 0.01
        self.weight_decay = 1e-4
        self.decay_rate = 0.01
        self.lr = 5e-5

        raw_data = load_dataset('data/load_dataset.py')
        dataset = CoNLL(self.task, raw_data)

        self.train_dataset = dataset.train_data
        self.dev_dataset = dataset.dev_data
        self.test_dataset = dataset.test_data
        self.ignore_columns = dataset.ignore_columns

        self.tokenizer = dataset.tokenizer
        self.data_collator = dataset.data_collator
        self.compute_metrics = dataset.compute_metrics
        self.bert_config = dataset.config

        self.model = BertForTokenClassification.from_pretrained(
            'bert-base-uncased',
            config=self.bert_config,
            revision='main',
        )

        self.train_loader = self.get_data_loader(self.train_dataset)
        self.dev_loader = self.get_data_loader(self.dev_dataset)
        self.test_loader = self.get_data_loader(self.test_dataset)
        self.max_seq_len = max([batch['labels'].shape[1] for _, batch in enumerate(self.dev_loader)])

    def get_sampler(self, dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        # Build the sampler.
        return RandomSampler(dataset, generator=generator)

    def get_data_loader(self, dataset: datasets.arrow_dataset.Dataset) -> DataLoader:
        dataset = dataset.remove_columns(self.ignore_columns)
        sampler = self.get_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )

    def get_optimizer(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]            
        optimizer_kwargs = {
            "betas": (self.adam_beta1, self.adam_beta2),
            "eps": self.adam_epsilon,
        }
        optimizer_kwargs["lr"] = self.lr            
        self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)

    def get_schedular(self):
        pass

    def pad_tensor(self, tensor: torch.Tensor, pad_index: int):
        max_size = self.max_seq_len
        old_size = tensor.shape
        new_size = list(old_size)
        new_size[1] = max_size
        new_tensor = tensor.new_zeros(tuple(new_size)) + pad_index
        new_tensor[:, : old_size[1]] = tensor
        return new_tensor

    def train(self):        
        self.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.decay_rate)
        pbar = tqdm(total=len(self.train_loader)*self.epoch)

        self.model.to(self.device)
        for epoch in range(self.epoch):
            # Train
            total_loss = 0
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                loss = output.loss
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                pbar.update(1)
            self.scheduler.step()

            # Evaluate
            self.model.eval()
            with torch.no_grad():
                labels, prediction = [], []
                for batch_idx, batch in enumerate(self.dev_loader):
                    batch = {k:v.to(self.device) for k,v in batch.items()}
                    output = self.model(**batch)
                    loss,logits = output.loss, output.logits
                    logits = self.pad_tensor(logits, -100)
                    prediction.append(logits)
                    batch_label = self.pad_tensor(batch['labels'], -100)
                    labels.append(batch_label)
                prediction = torch.cat(prediction)
                labels = torch.cat(labels)
                result = self.compute_metrics((np.array(prediction.cpu()), np.array(labels.cpu())))

                pbar.set_description(f'Train_loss: {total_loss}, Eval_F1: {result["f1"]}')

        pbar.close()
    

    def get_trainer_and_start(self):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        self.start()

    def start(self):
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics["train_samples"] = len(self.train_dataset)

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    train_api = Trainer_API()
    train_api.train()
    # train_api.get_trainer_and_start()