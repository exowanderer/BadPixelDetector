import pandas as pd

from glob import glob
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from time import time
from tqdm import tqdm


data_dir = 'data/NIRCam_Darks/CV3/'
amp_files = glob(data_dir + '*amp*.csv')

amps = {}
for kf,fname in tqdm(enumerate(amp_files)):
    key = 'amp{}'.format(kf)
    amps[key] = pd.read_csv(fname).values.T

start = time()

kmeans = {}
for key, amp in tqdm(amps.items(), total=len(amps)):
    kmeans[key] = KMeans(n_clusters=9, n_jobs=-1, verbose=True)
    kmeans[key].fit(amp)

print(time()-start)

time_stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")

joblib.dump(kmeans, 'nrc_darks_kmeans_by_amp_{}.joblib.save'.format(time_stamp)) # 20181024072942
