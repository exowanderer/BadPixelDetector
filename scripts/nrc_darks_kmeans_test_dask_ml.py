import pandas as pd

from dask_ml import cluster
from glob import glob
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from time import time

amp_files = glob('*amp*.csv')

try:
    amp0 = amp0
    print('Data Exists')
except:
    print('Loading Data')
    start = time()
    amp0 = pd.read_csv(amp_files[0]).values.T
    print('Loading Data took {} minutes'.format((time()-start)/60))

print('Starting SKLearn Process')
start = time()
kmeans = KMeans(n_clusters=9, n_jobs=-1)
kmeans.fit(amp0)
print('SKLaern Process took {} minutes'.format((time()-start)/60))

from dask.distributed import Client
from sklearn.externals.joblib import parallel_backend

client = Client()  # Connect to a Dask Cluster

print('Starting Dask Backend Process')
with parallel_backend('dask'):
    # Your normal scikit-learn code here
    start = time()
    kmeans = KMeans(n_clusters=9, n_jobs=-1)
    kmeans.fit(amp0)

print('Dask Backend Process took {} minutes'.format((time()-start)/60))

print('Starting Dask Default Process')
# Your normal scikit-learn code here
start = time()
kmeans = cluster.KMeans(n_clusters=9, n_jobs=-1)
kmeans.fit(amp0)

print('Dask Process took {} minutes'.format((time()-start)/60))
