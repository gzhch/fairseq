import json
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils

from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.models.roberta import RobertaClassificationHead

from fairseq.data import (
    Dictionary,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    SortDataset,
    data_utils,
    encoders,
    RawLabelDataset,
    BaseWrapperDataset
)

@register_task('squad2')
class SQuAD2Task(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--default_choices', default='', type=str)
        #max_positions

    def __init__(self, args, dictionary):
        super().__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = cls.dictionary = Dictionary.load(os.path.join(os.path.dirname(args.restore_file), 'dict.txt'))
        dictionary.add_symbol('<mask>')
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        path = self.args.data

        tokens = []
        starts = []
        ends = []
        unanswerables = []
        
        lengths = []
        
        for inp, p_mask, start, end, unanswerable in from_records(path):
            tokens.append(inp)
            lengths.append(len(inp))
            starts.append(start)
            ends.append(end)
            unanswerables.append(unanswerable)
            
        
        tokens = BaseWrapperDataset(tokens)
        starts = BaseWrapperDataset(np.array(starts, dtype=np.long))
        ends = BaseWrapperDataset(np.array(ends, dtype=np.long))
        lengths = np.array(lengths, dtype=np.long)
        unanswerables = BaseWrapperDataset(np.array(unanswerables, dtype=np.float32))


        print('| loaded {} batches from: {}'.format(len(lengths), path))

        shuffle = np.random.permutation(len(lengths))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'tokens': tokens,
                    'starts': starts,
                    'ends': ends,
                    'unanswerables': unanswerables,
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(tokens, reduce=True),
                },
                sizes=[lengths],
            ),
            sort_order=[
                shuffle,
            ],
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


from fairseq.data import BaseWrapperDataset
@register_task('squad2')
class SQuAD2Task(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--default_choices', default='', type=str)
        #max_positions

    def __init__(self, args, dictionary):
        super().__init__(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = cls.dictionary = Dictionary.load(os.path.join(os.path.dirname(args.restore_file), 'dict.txt'))
        dictionary.add_symbol('<mask>')
        print('| dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        path = self.args.data

        tokens = []
        starts = []
        ends = []
        unanswerables = []
        
        lengths = []
        
        for inp, p_mask, start, end, unanswerable in from_records(path):
            tokens.append(inp)
            lengths.append(len(inp))
            starts.append(start)
            ends.append(end)
            unanswerables.append(unanswerable)
            
        
        tokens = BaseWrapperDataset(tokens)
        starts = BaseWrapperDataset(np.array(starts, dtype=np.long))
        ends = BaseWrapperDataset(np.array(ends, dtype=np.long))
        lengths = np.array(lengths, dtype=np.long)
        unanswerables = BaseWrapperDataset(np.array(unanswerables, dtype=np.float32))


        print('| loaded {} batches from: {}'.format(len(lengths), path))

        shuffle = np.random.permutation(len(lengths))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    'id': IdDataset(),
                    'tokens': tokens,
                    'starts': starts,
                    'ends': ends,
                    'unanswerables': unanswerables,
                    'nsentences': NumSamplesDataset(),
                    'ntokens': NumelDataset(tokens, reduce=True),
                },
                sizes=[lengths],
            ),
            sort_order=[
                shuffle,
            ],
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

