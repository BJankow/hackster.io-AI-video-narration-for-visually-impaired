# standard library imports
from datetime import datetime
import logging
import sys

# 3rd party library imports
import numpy as np
from sklearn.cluster import DBSCAN

# local imports

FILENAME = "big_buck_bunny_1080p_h264_encoded-frames.npy"

PARALLELIZATION = 30

if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] (%(asctime)s) |%(name)s|: %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    all_encoded = np.load(FILENAME)

    t = datetime.now()
    clustered = DBSCAN(eps=8, min_samples=30, n_jobs=PARALLELIZATION).fit(all_encoded)
    logging.debug(f"Clustering with n_jobs={PARALLELIZATION} took {(datetime.now() - t).total_seconds()} [s]")

    # Noisy Samples: -1, Rest Groups: int
    labels = clustered.labels_
    str_labels = '|'.join(map(str, labels))
    pass

