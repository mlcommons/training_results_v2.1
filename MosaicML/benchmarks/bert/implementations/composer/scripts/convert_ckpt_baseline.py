####
# Example:
# python convert_ckpt_baseline.py --pt_checkpoint=/tmp/model.ckpt-28252.pt --save_checkpoint=/tmp/bert_large_baseline.pt
###
import torch
import logging
import argparse
import transformers
from composer.utils import reproducibility
from collections import OrderedDict

logger = logging.getLogger(__name__)


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
        default="/tmp/bert_large_baseline.pt",
        help="Path for saving Composer checkpoint",
    )
    return parser.parse_args()


def get_model(pretrained_checkpoint):
    config = transformers.AutoConfig.from_pretrained("bert-large-uncased")
    model = transformers.AutoModelForMaskedLM.from_config(config)
    orig_vocab_size = model.config.vocab_size
    if model.config.vocab_size % 8 != 0:
        model.config.vocab_size += 8 - (model.config.vocab_size % 8)
    model.resize_token_embeddings(model.config.vocab_size)
    model.eval()

    checkpoint = torch.load(pretrained_checkpoint)
    model_weights = checkpoint["model"]
    model_weights["cls.predictions.decoder.bias"] = model_weights[
        "cls.predictions.bias"
    ]
    missing_keys, unexpected_keys = model.load_state_dict(model_weights, strict=False)
    if len(missing_keys) > 0:
        logger.warning(
            f"Found these missing keys in the checkpoint: {', '.join(missing_keys)}"
        )
    if len(unexpected_keys) > 0:
        logger.warning(
            f"Found these unexpected keys in the checkpoint: {', '.join(unexpected_keys)}"
        )

    model.config.vocab_size = orig_vocab_size
    model.resize_token_embeddings(model.config.vocab_size)
    return model


def main():
    args = parse_arguments()

    model = get_model(args.pt_checkpoint)

    # append model to keys
    new_model_weights = OrderedDict()
    for key, value in model.state_dict().items():
        new_model_weights[f"model.{key}"] = value

    state_dict = {
        "state": {"model": new_model_weights},
        "rng": reproducibility.get_rng_state(),
    }

    # save checkpoint
    with open(args.save_checkpoint, "wb") as f:
        torch.save(state_dict, f)


if __name__ == "__main__":
    main()
