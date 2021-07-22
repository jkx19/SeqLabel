from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from model.prefix import BertForQuestionAnswering, BertPrefixModel
from transformers import AutoTokenizer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.trainer_pt_utils import get_parameter_names
import torch
from torch.optim import AdamW

from tqdm import tqdm
import argparse
import os
import sys

from data.squad_dataset import SQuAD


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
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
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
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

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
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
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training/validation file/test_file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."



class Train_API():
    
    def __init__(self, args) -> None:
        # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        self.batch_size = 16

        config = BertConfig.from_pretrained(
            'bert-base-uncased',
            revision='main',
        )
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            revision='main',
            use_fast=True,
        )
        config.num_labels = 2
        config.pre_seq_len = args.pre_seq_len
        config.mid_dim = args.mid_dim
        method = args.method
        if method == 'prefix':
            self.model = BertPrefixModel.from_pretrained(
                'bert-base-uncased',
                config=config,
                revision='main',
            )
        elif method == 'finetune':
            self.model = BertForQuestionAnswering.from_pretrained(
                'bert-base-uncased',
                config=config,
                revision='main',
            )
        dataset = SQuAD(tokenizer, self.batch_size)

        self.eval_example = dataset.eval_example
        self.eval_dataset = dataset.eval_dataset

        self.train_loader = dataset.train_loader
        self.eval_loader = dataset.eval_loader

        self.device = torch.device('cuda:0')
        self.batch_size = self.batch_size * torch.cuda.device_count()
        self.epoch = args.epoch
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.weight_decay = 0
        self.gamma = 0.99
        self.lr = args.lr

        self.compute_metric = dataset.compute_metric
        self.post_process_function = dataset.post_process_function


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

    def train(self):
        self.get_optimizer()
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.gamma)
        pbar = tqdm(total=(len(self.train_loader) + len(self.eval_loader))*self.epoch)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        best_dev_result = 0
        best_result = None
        for epoch in range(self.epoch):
            # Train
            total_loss = 0
            self.model.train()
            for batch_idx, batch in enumerate(self.train_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                loss = torch.sum(output.loss)
                # loss = output.loss
                total_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                pbar.update(1)
            self.scheduler.step()

            result = self.evaluate(pbar)
            eval_f1 = result['eval_f1']
            if eval_f1 > best_dev_result:
                best_dev_result = eval_f1
                best_result = result
            pbar.set_description(f'Train_loss: {total_loss:.1f}, Eval_F1: {eval_f1:.3f}')
        return result

    def evaluate(self, pbar: tqdm):
        self.model.eval()
        with torch.no_grad():
            start, end = [],[]
            for batch_idx, batch in enumerate(self.eval_loader):
                batch = {k:v.to(self.device) for k,v in batch.items()}
                output = self.model(**batch)
                start_logits, end_logits = output.start_logits, output.end_logits
                start.append(start_logits)
                end.append(end_logits)
                pbar.update(1)
            start_logits = np.array(torch.cat(start).cpu())
            end_logits = np.array(torch.cat(end).cpu())
        eval_preds = self.post_process_function(self.eval_example, self.eval_dataset, (start_logits, end_logits))
        metrics = self.compute_metric(eval_preds)
        for key in list(metrics.keys()):
                if not key.startswith(f"eval_"):
                    metrics[f"eval_{key}"] = metrics.pop(key)
        
        return metrics


def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pre_seq_len', type=int, default=10)
    parser.add_argument('--mid_dim', type=int, default=512)
    parser.add_argument('--method', type=str, choices=['finetune', 'prefix'], default='prefix')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = construct_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"      
    train_api = Train_API(args)
    result = train_api.train()
    sys.stdout = open('result.txt', 'a')
    print(result)
