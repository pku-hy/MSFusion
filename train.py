"""Dataset presets and a portable, validated entrypoint. Run python train.py --help."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import numpy as np
import pandas as pd

PRESETS = {
    'ETTh1': ('ETT-small/ETTh1.csv', 'ETTh1', 'h'),
    'ETTh2': ('ETT-small/ETTh2.csv', 'ETTh2', 'h'),
    'ETTm1': ('ETT-small/ETTm1.csv', 'ETTm1', '15min'),
    'ETTm2': ('ETT-small/ETTm2.csv', 'ETTm2', '15min'),
    'Weather': ('weather/weather.csv', 'custom', '10min'),
    'Electricity': ('electricity/electricity.csv', 'custom', 'h'),
    'Traffic': ('traffic/traffic.csv', 'custom', 'h'),
    'Solar': ('solar/solar_AL.txt', 'Solar', '10min'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=[*PRESETS, 'custom'], default='ETTh1')
    parser.add_argument('--dataset-dir', type=Path, default=Path(__file__).resolve().parent/'dataset')
    parser.add_argument('--file', type=Path, help='override preset path; required for custom CSV')
    parser.add_argument('--freq', help='custom CSV frequency, e.g. h, 15min, 10min')
    parser.add_argument('--pred-len', type=int, choices=[96,192,336,720], default=96)
    parser.add_argument('--seq-len', type=int, default=336)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--run-id', help='defaults to dataset + timestamp')
    parser.add_argument('--smoke', action='store_true', help='3 epochs total, one per stage, full data splits')
    parser.add_argument('--evaluate', action='store_true', help='evaluate saved checkpoints; supply original run-id and settings')
    parser.add_argument('--check-only', action='store_true', help='validate dataset and dependencies without training')
    args = parser.parse_args()
    if args.dataset == 'custom':
        if args.file is None or args.freq is None:
            parser.error('custom requires --file and --freq')
        data, freq = 'custom', args.freq
        path = args.file.resolve()
    else:
        rel, data, freq = PRESETS[args.dataset]
        path = (args.file or args.dataset_dir/rel).resolve()
        freq = args.freq or freq
    if not path.is_file():
        parser.error(f'Dataset not found: {path}. Place your dataset there or pass --file PATH.')
    if args.epochs < 3 or args.batch_size < 1 or args.seq_len < 4:
        parser.error('Require epochs >= 3, batch-size >= 1, seq-len >= 4')
    if data == 'Solar':
        values = np.loadtxt(path, delimiter=',')
        if values.ndim != 2:
            parser.error('Solar must be a comma-separated numeric matrix without a header')
        target = 'OT'
    else:
        frame = pd.read_csv(path)
        if frame.columns[0] != 'date' or len(frame.columns) < 2:
            parser.error('CSV must begin with date and contain numeric feature columns')
        dates = pd.to_datetime(frame['date'], errors='raise')
        if not dates.is_monotonic_increasing or dates.duplicated().any():
            parser.error('Dates must be strictly increasing, unique, and already sorted')
        values = frame.iloc[:,1:].to_numpy(dtype=float)
        target = str(frame.columns[-1])
    if not np.isfinite(values).all():
        parser.error('Dataset contains missing or non-finite values; preprocess before training')
    n, channels = values.shape
    if data.startswith('ETT'):
        unit = 4 if data.startswith('ETTm') else 1
        if n < 14400*unit:
            parser.error(f'{data} requires at least {14400*unit} rows for its fixed benchmark split')
        train_rows, val_rows, test_rows = 8640*unit, 2880*unit, 2880*unit
    else:
        train_rows, test_rows = int(n*.7), int(n*.2)
        val_rows = n-train_rows-test_rows
    if train_rows < args.seq_len+args.pred_len or min(val_rows,test_rows) < args.pred_len or train_rows-args.seq_len-args.pred_len+1 < args.batch_size:
        parser.error('Dataset splits are too short for these windows/batch size')
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; choose --device cpu or install a matching PyTorch/CUDA environment')
    print(f'Validated {path}: {n} rows, {channels} channels; split lengths {train_rows}/{val_rows}/{test_rows}', flush=True)
    if args.check_only:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        print('PASS: dataset and training imports')
        return
    if args.evaluate and not args.run_id:
        parser.error('--evaluate requires the original --run-id')
    run_id = args.run_id or f'{args.dataset}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    cmd = [sys.executable, str(Path(__file__).with_name('run.py')),
        '--task_name','long_term_forecast','--model','ms_fusion',
        '--is_training', '0' if args.evaluate else '1', '--model_id', run_id,
        '--data',data,'--root_path',str(path.parent),'--data_path',path.name,
        '--target',target,'--freq',freq,'--c_in',str(channels),'--c_out',str(channels),
        '--seq_len',str(args.seq_len),'--label_len','0','--pred_len',str(args.pred_len),
        '--train_epochs',str(3 if args.smoke else args.epochs),
        '--batch_size',str(args.batch_size),'--num_workers','0',
        '--use_gpu',str(args.device != 'cpu'),'--gpu',str(args.gpu)]
    print('Run ID:',run_id, flush=True)
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS','2')
    env.setdefault('MKL_NUM_THREADS','2')
    subprocess.run(cmd, cwd=Path(__file__).resolve().parent, env=env, check=True)

if __name__ == '__main__':
    main()
