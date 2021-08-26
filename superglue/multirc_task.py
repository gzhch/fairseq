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


@register_task("multirc")
class MultiRCTask(LegacyFairseqTask):
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

        if not hasattr(args, "max_positions"):
            self._max_positions = (
                args.max_source_positions,
                args.max_target_positions,
            )
        else:
            self._max_positions = args.max_positions
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
        question_ids = []
        labels = []

        def binarize(sent):
            sent_bpe = self.bpe.encode(sent).split()
            tokens = [self.vocab.index(token) for token in sent_bpe]
            return tokens + [self.source_dictionary.eos()]

        init_seq = [self.args.init_token]# if self.args.init_token is not None else []
        sep_seq = [self.args.separator_token]# if self.args.separator_token is not None else []

        with open(data_path, "r", encoding="utf8") as f:
            for l in f:
                line = json.loads(l)
                passage = binarize(line["passage"]["text"])
                passage_id = line["idx"]
                for question_dict in line["passage"]["questions"]:
                    question = binarize(question_dict["question"])
                    question_id = question_dict["idx"]
                    for answer_dict in question_dict["answers"]:
                        answer = binarize(answer_dict["text"])
                        answer_id = answer_dict["idx"]
                        
                        passage = passage[:self.max_positions() - len(question) - len(answer) - 3]
                        tokens = init_seq + passage + question + sep_seq + answer + sep_seq
                        tokens = torch.tensor(tokens, dtype=torch.int64)
                        
                        src_tokens.append(tokens)
                        question_ids.append(question_id)
                        if split != 'test':
                            labels.append(answer_dict["label"])

        print("nums of samples", len(src_tokens))
                
        src_lengths = np.array([len(t) for t in src_tokens])
        src_tokens = ListDataset(src_tokens, src_lengths)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": PadDataset(
                    src_tokens, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
                "src_lengths": NumelDataset(src_tokens, reduce=False),
            },
            "question_ids": RawLabelDataset(question_ids),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        if labels:
            dataset.update(target=RawLabelDataset(labels))

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

        model.register_classification_head(
            getattr(args, "classification_head_name", "sentence_classification_head"),
            num_classes=self.args.num_classes,
        )
        # model.classification_heads[
        #     getattr(args, "classification_head_name", "sentence_classification_head")
        # ] = RobertaClassificationHead(
        #     args.encoder_embed_dim * 3,
        #     args.encoder_embed_dim,
        #     args.num_classes,
        #     args.pooler_activation_fn,
        #     args.pooler_dropout,
        # )


        return model

    def max_positions(self):
        return self._max_positions

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        # stub to fool the fairseq criterion, which uses vocab's padding index
        return self.vocab