import modeling_bert_patched
from transformers import BertConfig, BertForPreTraining, load_tf_weights_in_bert
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument('--tf_checkpoint',
                        type=str,
                        default="/google_bert_data",
                        help="Path to directory containing TF checkpoint")
    parser.add_argument('--bert_config_path',
                        type=str,
                        default="/workspace/phase1",
                        help="Path bert_config.json is located in")
    parser.add_argument('--output_checkpoint', type=str,
                        default='./checkpoint.pt',
                        help="Path to output PyT checkpoint")

    return parser.parse_args()


def main():
    args = parse_arguments()

    config = BertConfig.from_json_file(args.bert_config_path)
    config.dense_seq_output = False

    model = BertForPreTraining(config)
    model = load_tf_weights_in_bert(model, config, args.tf_checkpoint)

    print(f"Save PyTorch model to {args.output_checkpoint}")
    torch.save(model.state_dict(), args.output_checkpoint)


if __name__ == "__main__":
    main()

