"""Integration checks for custom CSV and headerless Solar data, using synthetic values."""
from pathlib import Path
import tempfile
import subprocess
import sys
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    rng = np.random.default_rng(2021)
    values = rng.normal(size=(1200, 3))
    csv = folder/'custom.csv'
    frame = pd.DataFrame(values, columns=['a','b','c'])
    frame.insert(0,'date',pd.date_range('2020-01-01',periods=len(frame),freq='h'))
    frame.to_csv(csv,index=False)
    solar = folder/'solar.txt'
    np.savetxt(solar,values,delimiter=',')
    for dataset, path in [('custom',csv),('Solar',solar)]:
        cmd = [sys.executable,str(root/'train.py'),'--dataset',dataset,
            '--file',str(path),'--freq','h','--device','cpu','--smoke',
            '--seq-len','48','--batch-size','128','--run-id',f'verify_{dataset}']
        subprocess.run(cmd,check=True,cwd=root)
print('PASS: custom CSV and Solar three-stage integration')
