import pandas as pd

from glob import glob
from sklearn.externals import joblib
from time import time
from tqdm import tqdm
from umap import UMAP

amp_files = glob('*amp*.csv')

try:
    amps = amps
    print('Data Exists')
except:
    print('Loading Data')
    start = time()
    amps = {'amp{}'.format(k): pd.read_csv(fname).T for k, fname in tqdm(enumerate(amp_files), total=len(amp_files))}
    print('Loading data took {} minutes'.format(time() - start))

print('Starting UMAP Process')
start = time()

umaps = {key: UMAP() for key in amps.keys()}

tr_amps = {key: umap.fit_transform(amp) for (key, umap), amp in tqdm(zip(umaps.items(), amps.values()), total=len(amps))}

print('UMAP Process took {} minutes'.format(time() - start))

joblib.dump({'amps':amps, 'umaps':umaps}, 'nrc_darks_umaps_2D_fitted.joblib.save')
