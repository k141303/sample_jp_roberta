import os
import json
import argparse

from janome.tokenizer import Tokenizer # MeCabのPythonラッパー

import torch

from transformers import (
    RobertaModel,
    RobertaConfig
)

def load_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="model/roberta_base_wiki201221_janome_vocab_32000")
    parser.add_argument("--config_name", type=str, default="config.json")
    parser.add_argument("--vocab_name", type=str, default="vocab.json")
    parser.add_argument("--model_name", type=str, default="pytorch_model.bin")
    return parser.parse_args()

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == '__main__':
    args = load_arg()

    config = RobertaConfig.from_pretrained(
        os.path.join(args.model_dir, args.config_name)
    )
    model = RobertaModel.from_pretrained(
        os.path.join(args.model_dir, args.model_name),
        config=config
    )

    tokenizer = Tokenizer(wakati=True)
    tokens = list(tokenizer.tokenize("本日は晴天なり。"))
    tokens = ["<s>"] + tokens + ["</s>"]

    vocab = load_json(
        os.path.join(args.model_dir, args.vocab_name)
    )
    token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    outputs = model(input_ids = torch.tensor([token_ids]))
