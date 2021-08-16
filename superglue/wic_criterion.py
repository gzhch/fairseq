import math

import torch
import torch.nn.functional as F

import scipy.stats as stats
import numpy as np

from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("wic")
class WiCCriterion(FairseqCriterion):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--classification-head-name",
            default="sentence_classification_head",
            help="name of the ranking head to use",
        )

    def forward(self, model, sample, reduce=True):

        hiddens, _ = model(
            **sample["net_input"], features_only=True, return_all_hiddens=False
        )

        embeddings = []

        # first token [CLS]
        embeddings.append(hiddens[:, 0, :])

        # other tokens
        # net_input src_ranges range1/range2
        # shape of [batch, range_len] padded with 0
        for i in range(2):
            # [batch, range_len, hidden]
            index = (
                sample["net_input"]["src_ranges"][f"range{i+1}"]
                .unsqueeze(-1)
                .expand([-1, -1, hiddens.size(-1)])
            )
            # [batch, range_len, hidden]
            mask = index != 0
            # [batch, range_len, hidden]
            embedding = hiddens.gather(dim=1, index=index) * mask
            # [batch, hidden]
            embedding = embedding.sum(dim=1) / mask.sum(dim=1)
            embeddings.append(embedding)

        concat = torch.cat(embeddings, dim=1)

        # RobertaClassificationHead expects [batch, len, hidden]
        logits = model.classification_heads["sentence_classification_head"](
            concat.unsqueeze(1)
        )
        targets = sample["target_labels"]
        sample_size = targets.numel()

        loss = F.cross_entropy(logits.view(-1, 2), targets.view(-1), reduction="sum")
        
        tp = ((logits[:, 0] <= logits[:, 1]) & (targets == 1)).long().sum()
        fp = ((logits[:, 0] <= logits[:, 1]) & (targets == 0)).long().sum()
        fn = ((logits[:, 0] > logits[:, 1]) & (targets == 1)).long().sum()
        tn = ((logits[:, 0] > logits[:, 1]) & (targets == 0)).long().sum()
        assert (tp + fp + tn + fn) == targets.size(0), 'invalid size'
        
        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size,
        }

        _, preds = logits.max(dim=1)
        logging_output.update(ncorrect=(preds == targets).sum().item())
        logging_output.update(tp=utils.item(tp.data) if reduce else tp.data)
        logging_output.update(fp=utils.item(fp.data) if reduce else fp.data)
        logging_output.update(fn=utils.item(fn.data) if reduce else fn.data)
        logging_output.update(tn=utils.item(tn.data) if reduce else tn.data)
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1)

            tp_sum = float(sum(log.get('tp', 0) for log in logging_outputs))
            fp_sum = float(sum(log.get('fp', 0) for log in logging_outputs))
            fn_sum = float(sum(log.get('fn', 0) for log in logging_outputs))
            tn_sum = float(sum(log.get('tn', 0) for log in logging_outputs))
            if tp_sum + fp_sum + fn_sum + tn_sum > 0:
                assert tp_sum + fp_sum + fn_sum + tn_sum == sample_size, 'invalid size when aggregating'
                acc = (tp_sum + tn_sum) / sample_size
                tmp = 2 * tp_sum + fp_sum + fn_sum
                f1 = (2 * tp_sum) / tmp if tmp else 0
                tmp = (tp_sum + fp_sum) * (tp_sum + fn_sum) * (tn_sum + fp_sum) * (tn_sum + fn_sum)
                mcc = (tp_sum * tn_sum - fp_sum * fn_sum) / (tmp ** 0.5) if tmp else 0
                metrics.log_scalar('sample_size', sample_size)
                metrics.log_scalar('f1', f1)
                metrics.log_scalar('mcc', mcc)
                metrics.log_scalar('acc_f1', 0.5 * (acc + f1))
        if len(logging_outputs) > 0 and 'x' in logging_outputs[0]:
            x = np.concatenate([log.get('x', np.array([])) for log in logging_outputs])
            y = np.concatenate([log.get('y', np.array([])) for log in logging_outputs])
            pearson = stats.pearsonr(x, y)[0]
            spearman = stats.spearmanr(x, y)[0]
            metrics.log_scalar('pearson', pearson)
            metrics.log_scalar('spearman', spearman)
            metrics.log_scalar('pearson_spearman', 0.5 * (pearson + spearman))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

    # @staticmethod
    # def aggregate_logging_outputs(logging_outputs):
    #     """Aggregate logging outputs from data parallel training."""
    #     loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
    #     ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
    #     nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
    #     sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

    #     agg_output = {
    #         "loss": loss_sum / sample_size / math.log(2),
    #         "ntokens": ntokens,
    #         "nsentences": nsentences,
    #         "sample_size": sample_size,
    #     }

    #     if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
    #         ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
    #         agg_output.update(accuracy=ncorrect / nsentences)

    #     return agg_output
