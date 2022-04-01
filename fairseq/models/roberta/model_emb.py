import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import register_model

from .hub_interface import RobertaHubInterface
from .model import RobertaModel, RobertaEncoder


@register_model("ftemb")
class EmbModel(RobertaModel):
    # @classmethod
    # def hub_models(cls):
    #     return {
    #         "xlmr.base": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz",
    #         "xlmr.large": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz",
    #         "xlmr.xl": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xl.tar.gz",
    #         "xlmr.xxl": "http://dl.fbaipublicfiles.com/fairseq/models/xlmr/xlmr.xxl.tar.gz",
    #     }

    def __init__(self, args, encoder):
        super().__init__(args, encoder)
        
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder)


    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="sentencepiece",
        **kwargs
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return RobertaHubInterface(x["args"], x["task"], x["models"][0])


class EmbEncoder(RobertaEncoder):
    def __init__(self, args,  dictionary):
        super().__init(args, dictionary)

        self.ft_emb = super.build_embedding(len(dictionary), args.encoder_embed_dim, dictionary.pad())

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):

        
        return super().extract_features(src_tokens, return_all_hiddens, **kwargs)