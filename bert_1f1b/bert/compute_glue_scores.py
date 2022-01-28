# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from transformers.tokenization import BertTokenizer
from transformers.optimization import BertAdam


def run_evaluation(processor, output_mode, set_type):
    examples = processor.get_dev_examples(args.data_dir) if set_type == "dev" \
        else processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(
                    examples, processor.get_labels(), args.max_seq_length, tokenizer, output_mode)
    print(f" Running Evaluation on {set_type}")
    print("  Num examples = %d" % len(examples))
    print("  Batch size = %d" % args.eval_batch_size)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)

    all_label_ids = None
    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)
    
    data = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data.
    eval_sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    
    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids,
                        input_mask, labels=None)

        # Create eval loss and other metric required by the task.
        if set_type == "dev":
            if output_mode == "classification":
                if args.focal:
                    continue
                else:
                    loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(
                    logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                if args.fp16:
                    label_ids = label_ids.half()
                tmp_eval_loss = loss_fct(
                    logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)

    if set_type == "dev":
        eval_loss = eval_loss / nb_eval_steps
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss/nb_tr_steps if args.do_train else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss
        print("***** Evaluation results *****")
        for key in sorted(result.keys()):
            print("Epoch %s:  %s = %s" % (epoch_num,
                key, str(result[key])))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i) if set_type != "test" else line[0] 
            text_a = line[3]
            text_b = line[4]
            label = line[0] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0]) if set_type != "test" else line[0]
            text_a = line[8]
            text_b = line[9]
            label = line[-1] if set_type != "test" else "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_mismatched.tsv")),
            "test")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0 and set_type == "test":
                continue
            guid = "%s-%s" % (set_type, i) if set_type != "test" else line[0]
            text_a = line[3] if set_type != "test" else line[1]
            label = line[1] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i) if set_type != "test" else line[0]
            text_a = line[0] if set_type != "test" else line[1]
            label = line[1] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0]) if set_type != "test" else line[0]
            text_a = line[7]
            text_b = line[8]
            label = line[-1] if set_type != "test" else "0.0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0]) if set_type != "test" else line[0]
            try:
                text_a = line[3] if set_type != "test" else line[1]
                text_b = line[4] if set_type != "test" else line[2]
                label = line[5] if set_type != "test" else "0"
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0]) if set_type != "test" else line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
        
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0]) if set_type != "test" else line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "entailment"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0]) if set_type != "test" else line[0]
            text_a = line[1]
            text_b = line[2]
            label = line[-1] if set_type != "test" else "0"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3".
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2".
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict


def get_submodule_and_parameter_name(module, parameter_name):
    parameter_name_split = parameter_name.split(".")
    submodule = module
    for attribute_name in parameter_name_split[:-1]:
        submodule = getattr(submodule, attribute_name)
    return (submodule, parameter_name_split[-1])


def load_checkpoint(multistage_model, model, checkpoint_path, epoch, num_stages):
    if not os.path.isfile(os.path.join(checkpoint_path,
                                       f"checkpoint.0.pth.tar.epoch.{epoch}")):
        return

    num_modules = len(multistage_model)

    key_to_module_mapping = OrderedDict()
    all_stages_state_dict = OrderedDict()
    module_id = 0
    stage_id = 0
    for stage_id in range(num_stages):
        # Load the checkpoint associated with a stage.
        full_checkpoint_path = os.path.join(checkpoint_path,
                                            f"checkpoint.{stage_id}.pth.tar.epoch.{epoch}")
        checkpoint = torch.load(full_checkpoint_path,
                                map_location=torch.device('cpu'))

        # Iterate through all modules in stage_id's checkpoint.
        local_module_id = 0

        # Quit when checkpoints for all modules in full model are loaded.
        while module_id < num_modules:

            # Load checkpoint corresponding to different modules in our runtime.
            state_dict = checkpoint["state_dict"]
            state_dict_key = "module%d" % local_module_id

            if state_dict_key not in state_dict:
                break
            state_dict = checkpoint["state_dict"][state_dict_key]

            # Remove mask buffer. TODO: Might not need this?
            keys_to_delete = []
            for key in state_dict:
                if "mask" in key:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del state_dict[key]

            if checkpoint_from_distributed(state_dict):
                state_dict = unwrap_distributed(state_dict)

            # Collect all state_dicts in a single OrderedDict.
            for key in state_dict:
                all_stages_state_dict[(stage_id, local_module_id, key)] = \
                    state_dict[key]

            stage_module, _, _ = multistage_model[module_id]
            for key in state_dict:
                # key_to_module_mapping maps key (in state_dict) to the
                # torch.nn.Module wrapping the parameter and the name
                # of parameter (weight, bias, etc.).
                key_to_module_mapping[(stage_id, local_module_id, key)] = \
                    get_submodule_and_parameter_name(stage_module, key)

            local_module_id += 1
            module_id += 1

    model_state_dict = OrderedDict()
    for real_key in model.state_dict():
        (module, parameter_name) = get_submodule_and_parameter_name(
            model, real_key)
        # Find key in all_stages_state_dict that corresponds to real_key in
        # model's state_dict.
        for key in key_to_module_mapping:
            (module2, parameter_name2) = key_to_module_mapping[key]
            if parameter_name == parameter_name2 and str(module) == str(module2):
                break
        if parameter_name == parameter_name2 and str(module) == str(module2):
            model_state_dict[real_key] = all_stages_state_dict[key]
            del key_to_module_mapping[key]
            del all_stages_state_dict[key]
    return model_state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters.
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_config_path", type=str, required=True,
                        help="config to use.")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Filepath of directory with checkpoints.")
    parser.add_argument("--bert_model", default='bert-large-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument('--module', '-m', default=None,
                        help='name of module that contains model and tensor_shapes definition')
    parser.add_argument('--num_stages', default=-1, type=int,
                        help='number of stages in split GNMT model')
    parser.add_argument('--epoch', default=-1, type=int,
                        help='epoch that produced checkpoint')

    # Other parameters.
    parser.add_argument("--files_location", default=None, type=str, 
                        help="The directory in the blob storage which contains data and config files.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--final_learning_rate","--final_lr",
                        default=5e-6,
                        type=float,
                        help="The final learning rate for Adam.")
    parser.add_argument("--lr_decay_rate",
                        default=0.99,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--random',
                        default=False,
                        action='store_true',
                        help="Whether to fientune for random initialization")
    parser.add_argument('--focal',
                        default=False,
                        action='store_true',
                        help="Whether to use Focal Loss for finetuning.")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="Gamma parameter to be used in focal loss.")
    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
    }


    local_rank = 0
    if args.local_rank != -1:
        local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if local_rank == -1:
        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    world_size = 1
    if args.local_rank != -1:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        world_size = int(os.environ["WORLD_SIZE"])

    print("Starting download of tokenizer...")
    if args.local_rank > 0:
        torch.distributed.barrier()
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, cache_dir='baseline/',
        do_lower_case=args.do_lower_case)
    if args.local_rank == 0:
        torch.distributed.barrier()
    print("Done with download and loading of tokenizer...")

    if args.checkpoint_path is None:
        print("Starting download of pre-trained model...")
        if args.local_rank > 0:
            torch.distributed.barrier()
        model = BertForSequenceClassification.from_pretrained(
            args.bert_model,
            cache_dir='baseline/',
            num_labels=len(label_list))
        if args.local_rank == 0:
            torch.distributed.barrier()
        print("Done with download and loading of pre-trained model...")
    else:
        # Prepare model.
        config = BertConfig.from_json_file(args.bert_config_path)
        import importlib
        module = importlib.import_module(args.module)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        multistage_model = module.model(config, criterion)

        model = BertForSequenceClassification(
            config, num_labels=len(label_list))
        print("Loading checkpoint from %s for epoch %d using num_stages=%d" % (
            args.checkpoint_path, args.epoch, args.num_stages))
        model_state_dict = load_checkpoint(
            multistage_model, model, args.checkpoint_path,
            epoch=args.epoch, num_stages=args.num_stages)
        missing_keys, unexpected_keys = model.load_state_dict(
            model_state_dict, strict=False)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    if args.do_train:

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        print("***** Running training *****")
        print("  Num examples = %d" % len(train_examples))
        print("  Batch size = %d" % args.train_batch_size)
        print("  Num steps = %d" % num_train_optimization_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        for epoch_num in range(int(args.num_train_epochs)):
            model.train()
            if args.local_rank != -1:
                train_sampler.set_epoch(epoch_num)
            train_dataloader = DataLoader(
                train_data, sampler=train_sampler, batch_size=args.train_batch_size)
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            epoch_start_time = time.time()
            start_time = time.time()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    if args.focal:
                        loss_fct = FocalLoss(
                            class_num=num_labels, gamma=args.gamma, fp16=args.fp16)
                    else:
                        loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels),
                                    label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    if args.fp16:
                        label_ids = label_ids.half()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # Modify learning rate with special warm up BERT uses.
                        lr_this_step = args.learning_rate * \
                            warmup_linear(
                                global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step + 1) % 10 == 0:
                    print("Step [%d/%d] Time/iteration: %.3f seconds (%.3f seconds), Loss: %.3f (%.3f), Memory: %.3f GB (%.3f GB)" % (
                            step+1, len(train_dataloader),
                            time.time() - start_time,
                            (time.time() - epoch_start_time) / (step+1),
                            loss.item(), tr_loss / (step+1),
                            float(torch.cuda.memory_allocated()) / 10**9,
                            float(torch.cuda.memory_cached()) / 10**9))
                start_time = time.time()

            if args.do_eval:
                run_evaluation(processor, output_mode, "dev")
                if task_name == "mnli":
                    task_name = "mnli-mm"
                    run_evaluation(processors["mnli-mm"](), output_mode, "dev")
                    task_name = "mnli"
