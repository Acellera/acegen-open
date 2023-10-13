import json
import pandas as pd
import argparse
from pathlib import Path


def read_logs(logs_dir_path):
    """Read the logs from the log dir."""
    import ipdb; ipdb.set_trace()
    monitor_files = list(Path(logs_dir_path).glob("*monitor.json"))
    if not monitor_files:
        raise Exception(f"no monitor files found in {logs_dir_path}")
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
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


def analyse_logs(logs_dir_path):
    """Analise the logs from the log dir."""
    logs_df = read_logs(logs_dir_path)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Analyse de logs')
    parser.add_argument('--logs-path', type=str, help='Path to logs directory')
    args = parser.parse_args()
    analyse_logs(args.logs_path)

