import datetime
import time
import os

import pandas as pd

__all__ = ['savelog', 'convert_seconds_to_str',
           'filter_logfiles', 'concate_logfiles']


class savelog:
    ''' Saves training log to csv'''

    def __init__(self, directory, name, INCREMENTAL_UPDATE_TIME=30*60):
        self.file_path = os.path.join(
            directory, "{}_{:%Y-%m-%d_%H:%M:%S}.csv".format(name, datetime.datetime.now()))
        self.data = {}
        self.INCREMENTAL_UPDATE_TIME = INCREMENTAL_UPDATE_TIME
        self.last_update_time = time.time() - self.INCREMENTAL_UPDATE_TIME

    def record(self, step, value_dict):
        self.data[step] = value_dict
        if time.time() - self.last_update_time >= self.INCREMENTAL_UPDATE_TIME:
            self.last_update_time = time.time()
            self.save()

    def save(self):
        df = pd.DataFrame.from_dict(
            self.data, orient='index').to_csv(self.file_path)


def convert_seconds_to_str(seconds):
    return str(datetime.timedelta(seconds=seconds))


# some useful functions for concatenating log files
def filter_logfiles(files, startswith, endswith):
    results = []
    for f in files:
        if f.startswith(startswith) and f.endswith(endswith):
            results.append(f)
    return results


def concate_logfiles(files, timestep_name='Unnamed: 0'):
    files = sorted(files)

    dfs = []

    for f in files[::-1]:
        try:
            dfs.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            continue

    cleaned_dfs = [dfs[0]]

    for df in dfs[1:]:
        start_pos = cleaned_dfs[-1][timestep_name][0]
        cleaned_dfs.append(df.loc[df[timestep_name] < start_pos])

    cleaned_dfs = cleaned_dfs[::-1]
    return pd.concat(cleaned_dfs, axis=0)
