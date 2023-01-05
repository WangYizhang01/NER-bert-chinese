from .dataset import *

import argparse
import csv
import json
import logging
import os
import random
import sys
import numpy as np

class NerProcessor(DataProcessor):
    """Processor for the dataset."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels_from_datapath=''):
        # 从labels_from_datapath文件夹中train.txt获取标签集合，若地址为空则返回默认标签值
        if not labels_from_datapath:
            return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"], ["O", "PER", "ORG", "LOC"]
        else:
            labels = set(); span_label = set('O')
            for i,(sentence,label) in  enumerate(self._read_tsv(os.path.join(labels_from_datapath, "train.txt"))):
                labels.update(set(label))
                for l in label:
                    if '-' in l:
                        span_label.add(l.split('-')[1])
            return list(labels) + ["[CLS]", "[SEP]"], list(span_label)

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, span_label_list=None):
    """
    Loads a data file into a list of `InputBatch`s.
    if span_label_list is not none, 
    """

    label_map = {label : i for i, label in enumerate(label_list, 1)}
    if span_label_list:
        span_label_map = {label : i for i, label in enumerate(span_label_list, 1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        if span_label_list: entity_idx = convert_label_to_idx(labellist, max_seq_length)
        
        tokens = [] # 分词后的结果
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word) # 分词
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = [] # 在tokens中加入[CLS],[SEP]等标识符
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        
        label_start, label_end = None, None
        if span_label_list:
            label_start = [span_label_map['O']] * max_seq_length
            label_end = [span_label_map['O']] * max_seq_length
            for ent in entity_idx:
                ent, start, end = ent
                label_start[start] = span_label_map[ent]
                label_end[end] = span_label_map[ent]
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              label_start=label_start,
                              label_end=label_end))
    return features


def convert_label_to_idx(label, max_seq_length):
    """
    Args:
        label : ['O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O']
    Returns:
        entity_idx: [['LOC', 5, 6], ['LOC', 8, 9]]
    """
    if len(label) >= max_seq_length - 1:
        label = label[0:(max_seq_length - 2)]
    entity_idx = []
    i = 0
    while i < len(label):
        l = label[i]
        assert len(l) != 0
        if l[0] == 'B':
            tmp = [l[2:], i + 1] # 前面有[cls]，故 i+1
            i += 1
            while i < len(label):
                if label[i] == 'I-' + l[2:]:
                    i += 1
                else:
                    break
            tmp.append(i)
            entity_idx.append(tmp)
        i += 1
    return entity_idx
