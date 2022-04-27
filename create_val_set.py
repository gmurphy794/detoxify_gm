import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

def create_val_set(csv_file, val_fraction=.3):
    """Takes in a csv file path and creates a validation set
    out of it specified by val_fraction.
    """
    data_path = '../jigsaw_data/jigsaw-toxic-comment-classification-challenge/'
    dataset = pd.read_csv(data_path+csv_file)
    train_set, val_set = train_test_split(dataset, test_size=val_fraction)
    dataset.to_csv(data_path+"train_orig.csv",index=False)
    train_set.to_csv(data_path+'train.csv', index=False)
    val_set.to_csv(data_path+"val.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()
    create_val_set(args.csv, val_fraction=0.3)