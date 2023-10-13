import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def read_logs(logs_dir_path):
    """Read the logs from the log dir."""
    monitor_files = list(Path(logs_dir_path).glob("monitor*.csv"))
    if not monitor_files:
        raise Exception(f"no monitor files found in {logs_dir_path}")
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            firstline = fh.readline()
            if not firstline:
                continue
            assert firstline[0] == '#'
            header = json.loads(firstline[1:])
            df = pd.read_csv(fh, index_col=None)
            headers.append(header)
            df['t'] += header['t_start']  # TODO: ???
        dfs.append(df)
    df = pd.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)  # TODO: ???
    # df.headers = headers  # HACK to preserve backwards compatibility
    return df


def analyse_logs(logs_dir_path, N=20, rank_prop="r"):
    """Analise the logs from the log dir."""
    logs_df = read_logs(logs_dir_path)

    # Compute total steps
    total_steps = logs_df["l"].cumsum().iloc[-1]
    logs_df["total_steps"] = total_steps

    # Rank monitor files by property
    logs_df = logs_df.sort_values(rank_prop, ascending=False)

    # Remove duplicates
    logs_df.drop_duplicates(subset="molecule", inplace=True)

    # List top X molecules by score
    pd.options.display.max_colwidth = 300
    print(
        f"Here's a list of the top {N} molecules with the obtained score ({rank_prop}):"
    )
    print(
        logs_df[[rank_prop, "molecule"]].head(n=N)
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyse de logs')
    parser.add_argument('--logs-path', default="./logs", type=str, help='Path to logs directory')
    args = parser.parse_args()
    analyse_logs(args.logs_path)

