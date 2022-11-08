####
# Example:
# python convert_ckpt_methods.py --pt_checkpoint=/tmp/model.ckpt-28252.pt --save_checkpoint=/tmp/bert_large_methods.pt
###
import sys
import torch
import logging
import argparse
import transformers
from composer.utils import reproducibility
from collections import OrderedDict

logger = logging.getLogger(__name__)
sys.path.insert(0, "..")
from model import create_bert_unpadded_mlm


def parse_arguments():
    parser = argparse.ArgumentParser()
    # https://github.com/mlcommons/training_results_v2.0/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/convert_tf_checkpoint.py
    parser.add_argument(
        "--pt_checkpoint",
        type=str,
        default="/tmp/model.ckpt-28252.pt",
        help="Path to the NVidia converted Pytorch checkpoint",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=str,
        default="/tmp/bert_large_methods.pt",
        help="Path for saving Composer checkpoint",
    )
    return parser.parse_args()


def get_model(pretrained_checkpoint):
    # This will load the model weights from the given pretrained checkpoint
    m = create_bert_unpadded_mlm(
        use_pretrained=True,
        pretrained_model_name="bert-large-uncased",
        tokenizer_name="bert-large-uncased",
        pretrained_checkpoint=pretrained_checkpoint,
    )
    return m


def main():
    args = parse_arguments()

    m = get_model(args.pt_checkpoint)

    state_dict = {
        "state": {"model": m.state_dict()},
        "rng": reproducibility.get_rng_state(),
    }

    # save checkpoint
    with open(args.save_checkpoint, "wb") as f:
        torch.save(state_dict, f)


if __name__ == "__main__":
    main()
