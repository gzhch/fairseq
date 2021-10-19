import json
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import utils
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
    RawLabelDataset
)
from fairseq.models.roberta import RobertaClassificationHead
from fairseq.tasks import LegacyFairseqTask, register_task


@register_task("wic")
class WiCTask(LegacyFairseqTask):
    """Task to finetune RoBERTa for Winograd Schemas."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data", metavar="DIR", help="path to data directory; we load <split>.jsonl"
        )
        parser.add_argument(
            "--num-classes",
            type=int,
            default=-1,
            help="number of classes or regression targets",
        )
        parser.add_argument(
            "--init-token",
            type=int,
            default=None,
            help="add token at the beginning of each batch item",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default=None,
            help="add separator token between inputs",
        )
        parser.add_argument("--regression-target", action="store_true", default=False)
        

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.bpe = encoders.build_bpe(args)
        self.tokenizer = encoders.build_tokenizer(args)

        # self.init_index = self.vocab.add_symbol(args.init_token)
        # self.separator_index = self.vocab.add_symbol(args.separator_token)
        # # # hack to handle GPT-2 BPE, which includes leading spaces
        # if args.bpe == "gpt2":
        #     self.leading_space = True
        #     self.trailing_space = False
        # else:
        #     self.leading_space = False
        #     self.trailing_space = True

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == "wic", "Must set --criterion=wic"

        # load data and label dictionaries
        vocab = cls.load_dictionary(os.path.join(args.data, "dict.txt"))
        print("| dictionary: {} types".format(len(vocab)))

        return cls(args, vocab)


    def load_dataset(
        self, split, epoch=1, combine=False, data_path=None, return_only=False, **kwargs
    ):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # if split == 'valid':
        #     split = 'val'
        if data_path is None:
            data_path = os.path.join(self.args.data, split + ".jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError("Cannot find data: {}".format(data_path))

        src_tokens = []
        src_indices = [[], []]
        labels = []

        def binarize(sent, start, end, append_eos=True):
            if start > 0 and sent[start - 1].isspace():
                start = start - 1
            prefix = self.bpe.encode(sent[:start]).split()
            # the prefix space is included in the target
            target = self.bpe.encode(sent[start:end]).split()
            suffix = self.bpe.encode(sent[end:]).split()
            sent_bpe = prefix + target + suffix
            prefix_len = len(prefix)
            target_len = len(target)

            tokens = [self.vocab.index(token) for token in sent_bpe]
            if append_eos:
                tokens = tokens + [self.source_dictionary.eos()]
            return tokens, prefix_len, target_len

        init_seq = [self.args.init_token]# if self.args.init_token is not None else []
        sep_seq = [self.args.separator_token]# if self.args.separator_token is not None else []

        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                # json dict contains:
                # word, sentence1, sentence2, idx, start1, start2, end1, end2, label (true, false)
                data = json.loads(line)

                tokens1, offset1, len1 = binarize(
                    data["sentence1"], data["start1"], data["end1"]
                )
                tokens2, offset2, len2 = binarize(
                    data["sentence2"], data["start2"], data["end2"]
                )

                # [cls] sent1 [eos] [eos] sent2 [eos]
                tokens = init_seq + tokens1 + sep_seq + tokens2

                # types = [0] * (len(tokens) - len(tokens2)) + [1] * (len(tokens2))

                word1_offset = len(init_seq) + offset1
                word2_offset = len(init_seq) + len(tokens1) + len(sep_seq) + offset2

                tokens = torch.tensor(tokens, dtype=torch.int64)
                # types = torch.tensor(types, dtype=torch.int64)

                src_tokens.append(tokens)
                # src_types.append(types)

                for i, (offset, l) in enumerate(
                    [[word1_offset, len1], [word2_offset, len2]]
                ):

                    indice = torch.tensor(
                        [j for j in range(offset, offset + l)], dtype=torch.int64
                    )

                    src_indices[i].append(indice)

                if "label" in data:
                    labels.append(1 if data["label"] else 0)

        src_lengths = np.array([len(t) for t in src_tokens])
        src_tokens = ListDataset(src_tokens, src_lengths)

        for i, indice in enumerate(src_indices):
            assert all(
                i > 0 for j in indice for i in j
            ), "word token index should not be 0"
            lengths = np.array([len(t) for t in indice])
            src_indices[i] = ListDataset(indice, lengths)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": PadDataset(
                    src_tokens, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
                # "src_types": PadDataset(src_types, pad_idx=0, left_pad=False),
                "src_ranges": {
                    "range1": PadDataset(src_indices[0], pad_idx=0, left_pad=False),
                    "range2": PadDataset(src_indices[1], pad_idx=0, left_pad=False),
                },
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if labels:
            dataset.update(target_labels=RawLabelDataset(labels))

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_tokens.sizes])

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))
        dataset = SortDataset(
            nested_dataset,
            # shuffle
            sort_order=[shuffle],
        )

        if return_only:
            return dataset

        self.datasets[split] = dataset
        return self.datasets[split]


    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        # model.register_classification_head(
        #     getattr(args, "classification_head_name", "sentence_classification_head"),
        #     num_classes=self.args.num_classes,
        # )
        model.classification_heads[
            getattr(args, "classification_head_name", "sentence_classification_head")
        ] = RobertaClassificationHead(
            args.encoder_embed_dim * 3,
            args.encoder_embed_dim,
            args.num_classes,
            args.pooler_activation_fn,
            args.pooler_dropout,
        )


        return model
    
    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        # stub to fool the fairseq criterion, which uses vocab's padding index
        return self.vocab