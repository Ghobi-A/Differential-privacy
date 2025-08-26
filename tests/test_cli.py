import subprocess
import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt


def test_cli_creates_deterministic_output(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    input_path = tmp_path / 'data.csv'
    df.to_csv(input_path, index=False)

    out1 = tmp_path / 'out1.csv'
    cmd = [sys.executable, 'cli.py', '--input', str(input_path), '--output', str(out1), '--mechanism', 'laplace', '--random-state', '0']
    subprocess.run(cmd, check=True)
    assert out1.exists()
    res1 = pd.read_csv(out1)

    out2 = tmp_path / 'out2.csv'
    cmd[-2] = str(out2)  # replace output path
    subprocess.run(cmd, check=True)
    res2 = pd.read_csv(out2)

    pdt.assert_frame_equal(res1, res2)


def test_cli_randomised_response_deterministic(tmp_path):
    df = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['cat', 'dog', 'cat']})
    input_path = tmp_path / 'data.csv'
    df.to_csv(input_path, index=False)

    out1 = tmp_path / 'out1.csv'
    cmd = [
        sys.executable,
        'cli.py',
        '--input',
        str(input_path),
        '--output',
        str(out1),
        '--mechanism',
        'randomised-response',
        '--probability',
        '0.6',
        '--random-state',
        '0',
    ]
    subprocess.run(cmd, check=True)
    res1 = pd.read_csv(out1)

    out2 = tmp_path / 'out2.csv'
    cmd[-2] = str(out2)
    subprocess.run(cmd, check=True)
    res2 = pd.read_csv(out2)

    pdt.assert_frame_equal(res1, res2)
