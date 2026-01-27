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

# baseline KMeans with u=6 (as found)
u = 6
Cluster = KMeans(n_clusters=u, init='k-means++', random_state=51).fit(Xs)
centroids = Cluster.cluster_centers_
pred = Cluster.labels_

# helper
from collections import defaultdict

def compute_confusion(pred_label, unknown_Y, u, num_known):
    conf = np.zeros((u, max(num_known, int(np.max(unknown_Y))+1)))
    for i in range(len(unknown_Y)):
        ty = int(unknown_Y[i])
        conf[int(pred_label[i]), ty] += 1
    return conf


def compute_up_from_conf(conf, num_known):
    if conf.size == 0:
        return 0.0, []
    if conf.shape[1] > num_known:
        confusion_unknown = conf[:, num_known:]
    else:
        confusion_unknown = conf
    dominate = np.zeros(max(1, confusion_unknown.shape[1]))
    for row in range(confusion_unknown.shape[0]):
        row_sum = np.sum(confusion_unknown[row])
        if row_sum <= 0:
            continue
        for col in range(confusion_unknown.shape[1]):
            if confusion_unknown[row][col] >= row_sum * 0.5 and np.argmax(confusion_unknown[:, col]) == row:
                dominate[col] = confusion_unknown[row][col]
    unknown_acc_vals = []
    for clas in range(confusion_unknown.shape[1]):
        a = int(np.sum(unknown_Y == int(clas + num_known))) if conf.shape[1] > num_known else int(np.sum(unknown_Y == int(clas)))
        unknown_acc_vals.append(float(dominate[clas] / a) if a != 0 else 0.0)
    up = float(np.sum(dominate) / unknown_X.shape[0]) if unknown_X.shape[0] > 0 else float('nan')
    return up, unknown_acc_vals

# 1) Sweep dominance threshold
print('--- Dominance threshold sweep (replace 0.5 in dominance check) ---')
for thr in [0.5, 0.4, 0.33, 0.25]:
    conf = compute_confusion(pred, unknown_Y, u, num_known)
    # recompute up with thr
    if conf.shape[1] > num_known:
        confusion_unknown = conf[:, num_known:]
    else:
        confusion_unknown = conf
    dominate = np.zeros(max(1, confusion_unknown.shape[1]))
    for row in range(confusion_unknown.shape[0]):
        row_sum = np.sum(confusion_unknown[row])
        if row_sum <= 0:
            continue
        for col in range(confusion_unknown.shape[1]):
            if confusion_unknown[row][col] >= row_sum * thr and np.argmax(confusion_unknown[:, col]) == row:
                dominate[col] = confusion_unknown[row][col]
    unknown_acc_vals = []
    for clas in range(confusion_unknown.shape[1]):
        a = int(np.sum(unknown_Y == int(clas + num_known))) if conf.shape[1] > num_known else int(np.sum(unknown_Y == int(clas)))
        unknown_acc_vals.append(float(dominate[clas] / a) if a != 0 else 0.0)
    up = float(np.sum(dominate) / unknown_X.shape[0]) if unknown_X.shape[0] > 0 else float('nan')
    print('thr', thr, 'UP', up)

# 2) Merge clusters by centroid distance threshold
from scipy.spatial.distance import pdist, squareform
cdist = squareform(pdist(centroids))
# mean nonzero distance
nz = cdist + np.eye(cdist.shape[0]) * 1e9
mean_dist = np.mean(nz[nz < 1e8])
print('\nmean centroid dist (approx):', mean_dist)
for rel in [0.05, 0.1, 0.2]:
    thr = mean_dist * rel
    # union-find to merge centroids closer than thr
    parent = list(range(u))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for i in range(u):
        for j in range(i+1, u):
            if cdist[i,j] < thr:
                union(i,j)
    # remap labels
    remap = {i: find(i) for i in range(u)}
    new_ids = {}
    next_id = 0
    for i in range(u):
        r = find(i)
        if r not in new_ids:
            new_ids[r] = next_id
            next_id += 1
    mapped = np.array([new_ids[find(i)] for i in range(u)])
    pred_merged = mapped[pred]
    conf_merged = compute_confusion(pred_merged, unknown_Y, next_id, num_known)
    up_m, ua = compute_up_from_conf(conf_merged, num_known)
    print('merge rel', rel, '-> new clusters', next_id, 'UP', up_m)

print('\nDone')
