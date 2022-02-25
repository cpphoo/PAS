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

        target_df = []

        for i in dfs:
            if i.startswith('per_class') and not i.endswith('_all'):
                target_df.append((dfs[i]))


        compiled_result = target_df[0].copy()

        for i in range(1, len(target_df)):
            compiled_result += target_df[i]


        compiled_result /= len(target_df)
        compiled_result = construct_stats(compiled_result)


        with pd.ExcelWriter(target[:-5] + '_parent_metric_v2'+ '.xlsx', engine='xlsxwriter') as f:
            compiled_result.to_excel(f, sheet_name='per_parent_per_class')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, type=str, nargs='+', help='file that should be aggregated')
    args = parser.parse_args()

    compile_result(args)
