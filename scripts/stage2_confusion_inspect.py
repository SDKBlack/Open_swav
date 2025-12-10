#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

exp_dir = '/root/autodl-tmp/data/swav-main/swav-main/exp_wtnet_k2_pool16_red16_sl0.01_bl0.02/180'
X = np.load(os.path.join(exp_dir, 'test_X.npy'))
Y = np.load(os.path.join(exp_dir, 'test_Y.npy'))
label_hat = np.load(os.path.join(exp_dir, 'label_hat.npy'))

num_known = int(np.max(label_hat[label_hat>=0])) + 1 if np.any(label_hat>=0) else 0
unknown_idx = np.where(label_hat == -1)[0]
unknown_X = X[unknown_idx]
unknown_Y = Y[unknown_idx]

scaler = MinMaxScaler()
Xs = scaler.fit_transform(unknown_X)

# run KMeans for the chosen u (6) as in smoke test
u = 6
Cluster = KMeans(n_clusters=u, init='k-means++', random_state=51).fit(Xs)
pred_label = Cluster.labels_

# produce per-true-label mapping
true_unknowns = sorted(set(unknown_Y.tolist()))
print('num_known:', num_known)
print('true unknown labels:', true_unknowns)
for lab in true_unknowns:
    inds = np.where(unknown_Y == lab)[0]
    counts = np.bincount(pred_label[inds], minlength=u)
    total = counts.sum()
    top_cluster = np.argmax(counts)
    top_count = counts[top_cluster]
    print(f'label {int(lab):2d}: count {int(total):4d}, top_cluster {int(top_cluster)} top_frac {top_count/total:.3f}, counts {counts.tolist()}')

# For each cluster, show which true labels contribute most
print('\nPer-cluster dominant true labels:')
for c in range(u):
    inds = np.where(pred_label == c)[0]
    if len(inds) == 0:
        print('cluster', c, 'empty')
        continue
    labs, ct = np.unique(unknown_Y[inds], return_counts=True)
    sorted_pairs = sorted(zip(labs, ct), key=lambda x: -x[1])
    print('cluster', c, 'size', len(inds), 'top contributors:', [(int(li), int(ci)) for li,ci in sorted_pairs[:5]])
