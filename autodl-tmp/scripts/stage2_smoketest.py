#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn import metrics as skm
from sklearn.decomposition import PCA
import os

exp_dir = '/root/autodl-tmp/data/swav-main/swav-main/exp_wtnet_k2_pool16_red16_sl0.01_bl0.02/180'

def load_arrays():
    X = np.load(os.path.join(exp_dir, 'test_X.npy'))
    Y = np.load(os.path.join(exp_dir, 'test_Y.npy'))
    label_hat = np.load(os.path.join(exp_dir, 'label_hat.npy'))
    theta = np.load(os.path.join(exp_dir, 'theta.npy'))
    return X, Y, label_hat, theta


def infer_num_known(label_hat, Y):
    if np.any(label_hat >= 0):
        return int(np.max(label_hat)) + 1
    else:
        # fallback: assume labels < median are known? use min continuous heuristic
        return int(max(1, int(np.min(Y))))


def compute_confusion(pred_label, unknown_Y, num_known):
    u = np.max(pred_label) + 1
    num_unknown = int(np.max(unknown_Y) - num_known + 1) if np.max(unknown_Y) >= num_known else 0
    conf = np.zeros((u, max(num_known, num_known + num_unknown)))
    for i in range(len(unknown_Y)):
        ty = int(unknown_Y[i])
        ty_idx = ty if ty >= 0 else 0
        if ty_idx >= conf.shape[1]:
            ty_idx = conf.shape[1] - 1
        conf[int(pred_label[i]), ty_idx] += 1
    return conf


def orig_stage2(unknown_X, unknown_Y, num_known):
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(unknown_X)
    DB = []
    candidates = list(range(2, 15))
    labels_store = {}
    for ui in candidates:
        try:
            Cluster = KMeans(n_clusters=ui, init='k-means++', random_state=51).fit(Xs)
            pre_label = Cluster.labels_
            db = float(skm.davies_bouldin_score(Xs, pre_label)) if Xs.shape[0] > ui else float('inf')
        except Exception:
            pre_label = np.zeros(Xs.shape[0], dtype=int)
            db = float('inf')
        DB.append(db)
        labels_store[ui] = pre_label
    DB = np.array(DB)
    best_idx = int(np.nanargmin(DB))
    u = candidates[best_idx]
    pred_label = labels_store[u]
    conf = compute_confusion(pred_label, unknown_Y, num_known)
    # focus unknown columns
    num_unknown = int(np.max(unknown_Y) - num_known + 1) if np.max(unknown_Y) >= num_known else 0
    if num_unknown > 0 and conf.shape[1] > num_known:
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
        a = int(np.sum(unknown_Y == int(clas + num_known)))
        unknown_acc_vals.append(float(dominate[clas] / a) if a != 0 else 0.0)
    up = float(np.sum(dominate) / unknown_X.shape[0]) if unknown_X.shape[0] > 0 else float('nan')
    return {'u': int(u), 'unknown_acc': unknown_acc_vals, 'mean_unknown_acc': float(np.mean(unknown_acc_vals)) if len(unknown_acc_vals)>0 else float('nan'), 'UP': up, 'confusion': conf}


def improved_stage2(unknown_X, unknown_Y, num_known):
    X = unknown_X.copy()
    scaler = StandardScaler()
    try:
        Xs = scaler.fit_transform(X)
    except Exception:
        Xs = X.copy()
    n_samples, n_features = Xs.shape
    if n_features > 50 and n_samples > 50:
        try:
            pca = PCA(n_components=min(50, n_features, max(1, n_samples-1)), svd_solver='auto', random_state=51)
            Xs = pca.fit_transform(Xs)
        except Exception:
            pass
    try:
        Xs = normalize(Xs, norm='l2')
    except Exception:
        pass
    DB_vals = []
    SIL_vals = []
    candidates = list(range(2, min(20, max(3, Xs.shape[0]))))
    labels_store = {}
    for ui in candidates:
        try:
            Cluster = KMeans(n_clusters=ui, init='k-means++', n_init=30, random_state=51).fit(Xs)
            pre_label = Cluster.labels_
            db = skm.davies_bouldin_score(Xs, pre_label) if Xs.shape[0] > ui else np.inf
            sil = skm.silhouette_score(Xs, pre_label) if Xs.shape[0] > ui and ui > 1 else -1.0
        except Exception:
            pre_label = np.zeros(Xs.shape[0], dtype=int)
            db = np.inf
            sil = -1.0
        DB_vals.append(db)
        SIL_vals.append(sil)
        labels_store[ui] = pre_label
    DB_vals = np.array(DB_vals)
    SIL_vals = np.array(SIL_vals)
    use_silhouette = np.isfinite(SIL_vals).sum() > 0 and Xs.shape[0] > 30
    best_idx_in_filtered = None
    if use_silhouette and np.nanmax(SIL_vals) > -0.5:
        max_sil = float(np.nanmax(SIL_vals))
        threshold = max_sil * 0.9
        candidates_meet = [i for i, s in enumerate(SIL_vals) if s >= threshold]
        if len(candidates_meet) > 0:
            best_idx_in_filtered = candidates_meet[0]
        else:
            best_idx_in_filtered = int(np.nanargmax(SIL_vals))
        best_k = candidates[best_idx_in_filtered]
    else:
        best_k = candidates[int(np.nanargmin(DB_vals))]
    pred_label = labels_store[best_k]
    conf = compute_confusion(pred_label, unknown_Y, num_known)
    num_unknown = int(np.max(unknown_Y) - num_known + 1) if np.max(unknown_Y) >= num_known else 0
    if num_unknown > 0 and conf.shape[1] > num_known:
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
        a = int(np.sum(unknown_Y == int(clas + num_known)))
        unknown_acc_vals.append(float(dominate[clas] / a) if a != 0 else 0.0)
    up = float(np.sum(dominate) / unknown_X.shape[0]) if unknown_X.shape[0] > 0 else float('nan')
    return {'u': int(best_k), 'unknown_acc': unknown_acc_vals, 'mean_unknown_acc': float(np.mean(unknown_acc_vals)) if len(unknown_acc_vals)>0 else float('nan'), 'UP': up, 'confusion': conf}


def pretty_print(res, title):
    print('---', title, '---')
    print('u:', res['u'])
    print('UP:', res['UP'])
    print('mean_unknown_acc:', res.get('mean_unknown_acc'))
    print('unknown_acc per class:', res.get('unknown_acc'))
    print('confusion shape:', res['confusion'].shape)


if __name__ == '__main__':
    X, Y, label_hat, theta = load_arrays()
    num_known = infer_num_known(label_hat, Y)
    unknown_idx = np.where(label_hat == -1)[0]
    unknown_X = X[unknown_idx]
    unknown_Y = Y[unknown_idx]
    print('num_known inferred:', num_known)
    print('unknown count:', unknown_X.shape[0])
    r1 = orig_stage2(unknown_X, unknown_Y, num_known)
    pretty_print(r1, 'Original Stage2 (MinMax + KMeans DB)')
    r2 = improved_stage2(unknown_X, unknown_Y, num_known)
    pretty_print(r2, 'Improved Stage2 (Std + PCA + L2 + KMeans Sil/DB)')
    # show which true unknown classes have very low acc
    print('\nTrue unknown class totals:')
    unknown_labels_sorted = sorted(list(set(unknown_Y.tolist())))
    for lab in unknown_labels_sorted:
        print('label', int(lab), 'count', int(np.sum(unknown_Y == lab)))

    print('\nDone')
