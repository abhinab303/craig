import pdb
import numpy as np
import pandas as pd
import matplotlib
from sklearn.manifold import TSNE

stats = np.load("./tmp/mnist_ep10_stats.npz")
subsets = np.load("./tmp/mnist_ep10_ss.npz")
indices = subsets['indices']
weights = subsets['weight']
feature = stats['features']
P = stats['logits']
Y = stats['Y_train']

X = TSNE(n_components=2,n_jobs=4).fit_transform(feature)
pdb.set_trace()

