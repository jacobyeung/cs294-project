import numpy as np
import pandas as pd
import json
import argparse


def load(root):
    df = pd.read_csv(root, sep='\t')
    return df.T


def get_genes(fname):
    df = pd.read_csv(fname, dtype=str)
    gene_id = df.iloc[:, 0].astype(str)
    return pd.unique(gene_id)


def main():
    root = args.dfname
    df = load(root)[2:-1]
    test = args.testname
    gene_id = get_genes(test)
    smallest_key = len(min(gene_id, key=len))
    bool_pd = np.isin(df.columns.str[-smallest_key:], gene_id)
    df = df.T[bool_pd].T
    # print(df.iloc[1, :])
    med = df.median(axis=1)
    print(med)
    df['median'] = med
    js_df = df.to_json(path_or_buf="rpkm.json",
                       force_ascii=False, orient="index", indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepChrome")
    parser.add_argument('--dfname', help='input file', type=str)
    parser.add_argument('--testname', help='test file', type=str)
    args = parser.parse_args()
    main()
