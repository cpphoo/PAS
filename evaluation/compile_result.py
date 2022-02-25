import pandas as pd
import numpy as np

import argparse

def construct_stats(df):
    
    mean = df.mean()
    std = df.std()

    confi_interval = std * 1.96 / np.sqrt(len(df))
    aggregated = pd.concat([mean, std, confi_interval], axis=1)
    aggregated.columns = ['mean', 'std', '95%CI']
    return aggregated



def compile_result(args):
    for target in args.file:
        dfs = pd.read_excel(target, sheet_name=None, index_col=0)

        compiled_result = {}

        for i in dfs:
            compiled_result[i] = construct_stats(dfs[i])

        with pd.ExcelWriter(target[:-5] + '_aggregated'+ '.xlsx', engine='xlsxwriter') as f:
            for i in compiled_result:
                compiled_result[i].to_excel(f, sheet_name=i)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str, nargs='+', help='file that should be aggregated')
    args = parser.parse_args()

    compile_result(args)
