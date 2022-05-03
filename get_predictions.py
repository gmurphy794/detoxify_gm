import argparse
import os
import torch
import pandas as pd
from src.finetuning_utils import model_init
import numpy as np
import json


def load_text(csv_file):
    """Checks input_obj is either the path to a txt file or a text string.
    If input_obj is a txt file it returns a list of strings."""

    df = pd.read_csv(csv_file)
    return df


def run(input_obj, config, dest_file, from_ckpt, device="cpu"):
    """Loads model from checkpoint or from model name and runs inference on the input_obj.
    Displays results as a pandas DataFrame object.
    If a dest_file is given, it saves the results to a txt file.
    """
    df = load_text(input_obj)
    base_path = 'saved/checkpoints/'
    checkpoint_path = base_path + from_ckpt
    config = json.load(open(args.config))
    model = model_init(config, checkpoint_path)
    model.eval()
    model.to(device)

    preds = []  
    with torch.no_grad():
        i = 0
        while i < df.shape[0]:
            try:
                text = list(df['comment_for_evaluation'][i:i+10].values)
            except:
                text = list(df['comment_for_evaluation'][i:].values)
            pred = model.forward(text)
            scores = torch.sigmoid(pred).cpu().detach().numpy()
            preds.append(scores.round(5))
            i+=10
    
    preds = np.concatenate(preds)
    column_names = [
                "toxicity",
                "severe_toxicity",
                "obscene",
                "threat",
                "insult",
                "identity_attack"]
    
    out = pd.DataFrame(preds, columns = column_names)
    out.to_csv(dest_file, index=False)


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        help="text, list of strings, or txt file",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to load the model on",
    )

    parser.add_argument(
        "--from_ckpt_path",
        default=None,
        type=str,
        help="Option to load from the checkpoint path (default: False)",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        type=str,
        help="destination path to output model results to (default: None)",
    )

    args = parser.parse_args()


    run(
        args.input,
        args.config,
        args.save_to,
        args.from_ckpt_path,
        device=args.device,
    )
