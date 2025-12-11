"""
Compute Stage-1 and Stage-2 open-set metrics for several clustering methods using the
same logic as `main_swav.py`'s _compute_stage2_summary and metrics_stage_1.

Usage:
  python3 scripts/stage2_compute_open_set_metrics.py --exp_dir /path/to/exp_dir

Outputs:
  - <exp_dir>/stage2_open_set_metrics_by_method.json
  - prints a concise table

This script does NOT modify `main_swav.py`.
"""
import os
import json
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture


def outlier_check(dist_list):
    # replicate a simple heuristic: use mean + 3*std as threshold (fallback)
    try:
        d = np.asarray(dist_list)
        med = np.median(d)
        mad = np.median(np.abs(d - med))
        # robust threshold: median + 3 * MAD (scaled)
        return float(med + 3.0 * (mad if mad > 1e-12 else (d.std() if d.size>0 else 0.0)))
    except Exception:
        return float(np.inf)


def metrics_stage_1(true_label, predict_label, num_known_local):
    import numpy as _np
    # TKR: fraction of known correctly predicted (pred != -1 and true != -1)
    a = _np.sum(_np.logical_and(predict_label != -1, true_label != -1))
    b = _np.sum(true_label != -1)
    tkr = float(a / b) if b != 0 else float('nan')
    # TUR: true unknown recall (unknown correctly predicted as unknown)
    a = _np.sum(_np.logical_and(true_label == -1, predict_label == -1))
    b = _np.sum(true_label == -1)
    tur = float(a / b) if b != 0 else float('nan')
    # KP: known precision (among predicted known, fraction correct)
    a = _np.sum(true_label[true_label != -1] == predict_label[true_label != -1])
    b = _np.sum(predict_label != -1)
    kp = float(a / b) if b != 0 else 0.0
    # FKR: false known rate = fraction of unknown predicted as known
    a = _np.sum(predict_label[true_label == -1] != -1)
    b = _np.sum(true_label == -1)
    fkr = float(a / b) if b != 0 else float('nan')
    accuracy = []
    for ii in range(num_known_local):
        a = _np.sum(_np.logical_and(true_label == ii, predict_label == ii))
        b = _np.sum(true_label == ii)
        if b != 0:
            accuracy.append(float(a / b))
        else:
            accuracy.append(-1.0)
    return tkr, tur, kp, fkr, accuracy


def compute_stage2_from_preds(test_X, test_Y, label_hat_stage1, theta, preds_unknown, num_known, stage2_dominance=0.4, force_clusterize=False):
    # preds_unknown: length = n_unknown, cluster labels 0..u-1 (DBSCAN may have -1 for noise)
    unknown_idx = np.where(label_hat_stage1 == -1)[0]
    unknown_X_local = test_X[unknown_idx]
    unknown_Y_local = test_Y[unknown_idx]

    stage2 = {}
    if unknown_X_local.shape[0] == 0:
        stage2['note'] = 'No unknown samples predicted by stage1.'
        return stage2

    # normalize true labels: labels >= num_known become -1
    test_Y_normalized = test_Y.copy()
    try:
        test_Y_normalized[test_Y_normalized >= num_known] = -1
    except Exception:
        pass

    # compute Mahalanobis-like outlier check for u=1 case (same as main_swav)
    try:
        samples = unknown_X_local.copy()
        covariance_mat = np.cov(samples, rowvar=False, bias=True)
        matrix = np.linalg.pinv(covariance_mat)
        centers = np.mean(samples, axis=0)
        x = samples - centers[None, :]
        try:
            dist_list = np.sqrt(np.einsum('ij,jk,ik->i', x, matrix, x))
        except Exception:
            dist_list = np.linalg.norm(x, axis=1)
        theta_u1 = outlier_check(dist_list)
    except Exception:
        theta_u1 = float('inf')

    # check u=1 compactness (unless user forces clustering)
    if (not force_clusterize) and (theta_u1 <= np.max(theta) if theta.size > 0 else False):
        a = int(np.sum(np.logical_and(test_Y_normalized == -1, label_hat_stage1 == -1)))
        b = int(np.sum(test_Y_normalized == -1))
        unknown_acc_val = float(a / b) if b != 0 else float('nan')
        c = unknown_X_local.shape[0]
        UP_val = float(a / c) if c != 0 else float('nan')
        # For u=1 case, we return scalar unknown_acc (recall over all true unknowns)
        # and UP (precision among predicted-as-unknown). Also include counts for clarity.
        stage2.update({
            'u': 1,
            'unknown_acc': unknown_acc_val,
            'UP': UP_val,
            'n_predicted_unknowns': int(c),
            'n_true_unknowns': int(b),
            'mean_unknown_acc': float(unknown_acc_val) if not np.isnan(unknown_acc_val) else float('nan')
        })
        return stage2

    # Otherwise use provided preds_unknown
    pred_label = np.array(preds_unknown)
    # Remap cluster labels to consecutive [0..u-1] treating DBSCAN -1 as a cluster index as well
    unique_clusters = np.unique(pred_label)
    # if DBSCAN noise label -1 present, include it as a row as main_swav would (they used KMeans only)
    # But we'll treat -1 as a valid cluster index for counting
    # Build confusion matrix rows = number unique cluster ids, cols = num_known + num_unknown
    # Compute num_unknown from test_Y
    num_unknown = int(np.max(test_Y) - num_known + 1) if np.max(test_Y) >= num_known else 0
    cols = num_known + max(0, num_unknown)
    u = len(unique_clusters)
    cluster_id_to_row = {cid: i for i, cid in enumerate(unique_clusters)}
    confusion_mat = np.zeros((u, cols))
    for xi in range(unknown_X_local.shape[0]):
        ty = int(unknown_Y_local[xi]) if unknown_Y_local.size > 0 else 0
        ty_idx = 0 if ty < 0 else int(ty)
        if ty_idx >= confusion_mat.shape[1]:
            ty_idx = confusion_mat.shape[1] - 1
        row = cluster_id_to_row[pred_label[xi]]
        confusion_mat[row][ty_idx] += 1

    if num_unknown > 0 and confusion_mat.shape[1] > num_known:
        confusion_unknown = confusion_mat[:, num_known:]
    else:
        confusion_unknown = confusion_mat

    dominate_sample = np.zeros(max(1, confusion_unknown.shape[1]))
    for row in range(confusion_unknown.shape[0]):
        row_sum = np.sum(confusion_unknown[row])
        if row_sum <= 0:
            continue
        for col in range(confusion_unknown.shape[1]):
            if confusion_unknown[row][col] >= row_sum * stage2_dominance and np.argmax(confusion_unknown[:, col]) == row:
                dominate_sample[col] = confusion_unknown[row][col]

    unknown_acc_vals = []
    for clas in range(confusion_unknown.shape[1]):
        a = int(np.sum(test_Y == int(clas + num_known)))
        unknown_acc_vals.append(float(dominate_sample[clas] / a) if a != 0 else 0.0)
    up = float(np.sum(dominate_sample) / unknown_X_local.shape[0]) if unknown_X_local.shape[0] > 0 else float('nan')
    stage2.update({'u': int(u), 'unknown_acc': unknown_acc_vals, 'mean_unknown_acc': float(np.mean(unknown_acc_vals)) if len(unknown_acc_vals) > 0 else float('nan'), 'UP': up})
    return stage2


def cluster_and_eval(exp_dir, method, k_min=2, k_max=14, dbscan_eps_list=[0.5,1.0,1.5,2.0], stage2_dominance=0.4, force_clusterize=False):
    # load necessary artifacts
    test_X = np.load(os.path.join(exp_dir, 'test_X.npy'))
    test_Y = np.load(os.path.join(exp_dir, 'test_Y.npy'))
    label_hat = np.load(os.path.join(exp_dir, 'label_hat.npy'))
    theta = np.load(os.path.join(exp_dir, 'theta.npy')) if os.path.exists(os.path.join(exp_dir, 'theta.npy')) else np.zeros((1,))
    centers_path = os.path.join(exp_dir, 'centers.npy')
    if os.path.exists(centers_path):
        centers = np.load(centers_path)
        num_known = centers.shape[0]
    else:
        # fallback inference
        num_known = int(np.max(test_Y[test_Y < 1e9]) + 1) if test_Y.size>0 else 0

    # stage1 metrics
    # normalize test_Y true labels as in main_swav
    test_Y_normalized = test_Y.copy()
    for xi in range(test_Y_normalized.shape[0]):
        if test_Y_normalized[xi] >= num_known:
            test_Y_normalized[xi] = -1
    # recompute stage1 label_hat as main_swav did (we have label_hat already but recompute for safety?)
    # We'll use provided label_hat for stage1 metrics
    tkr, tur, kp, fkr, accuracy = metrics_stage_1(test_Y_normalized, label_hat, num_known)

    unknown_idx = np.where(label_hat == -1)[0]
    X_unknown = test_X[unknown_idx]
    y_unknown = test_Y[unknown_idx] if test_Y is not None else None

    result = {'method': method, 'num_known': int(num_known), 'n_unknown': int(X_unknown.shape[0]), 'stage1': {'tkr': tkr, 'tur': tur, 'kp': kp, 'fkr': fkr}}

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_unknown) if X_unknown.shape[0] > 0 else X_unknown

    if method == 'kmeans':
        # search k by DB like main_swav
        import sklearn.metrics as skm
        DB = []
        candidates = list(range(k_min, k_max+1))
        best_idx = None
        best_k = None
        for ui in candidates:
            try:
                Cluster = KMeans(n_clusters=ui, init='k-means++', random_state=51).fit(Xs)
                pre_label = Cluster.labels_
                db = float(skm.davies_bouldin_score(Xs, pre_label)) if Xs.shape[0] > ui else float('inf')
            except Exception:
                db = float('inf')
            DB.append(db)
        try:
            best_idx = int(np.nanargmin(np.array(DB)))
            best_k = candidates[best_idx]
        except Exception:
            best_k = k_min
        Cluster = KMeans(n_clusters=best_k, init='k-means++', random_state=51).fit(Xs)
        preds = Cluster.labels_
        stage2 = compute_stage2_from_preds(test_X, test_Y, label_hat, theta, preds, num_known, stage2_dominance, force_clusterize=force_clusterize)
        result['params'] = {'k': int(best_k)}
        result['stage2'] = stage2
    elif method == 'agglomerative':
        # search k via DB
        import sklearn.metrics as skm
        DB = []
        candidates = list(range(k_min, k_max+1))
        best_k = k_min
        for ui in candidates:
            try:
                Cluster = AgglomerativeClustering(n_clusters=ui, linkage='ward').fit(Xs)
                pre_label = Cluster.labels_
                db = float(skm.davies_bouldin_score(Xs, pre_label)) if Xs.shape[0] > ui else float('inf')
            except Exception:
                db = float('inf')
            DB.append(db)
        try:
            best_k = candidates[int(np.nanargmin(np.array(DB)))]
        except Exception:
            best_k = k_min
        Cluster = AgglomerativeClustering(n_clusters=best_k, linkage='ward').fit(Xs)
        preds = Cluster.labels_
        result['params'] = {'k': int(best_k)}
        result['stage2'] = compute_stage2_from_preds(test_X, test_Y, label_hat, theta, preds, num_known, stage2_dominance, force_clusterize=force_clusterize)
    elif method == 'gmm':
        import sklearn.metrics as skm
        DB = []
        candidates = list(range(k_min, k_max+1))
        best_k = k_min
        best_preds = None
        for ui in candidates:
            try:
                model = GaussianMixture(n_components=ui, covariance_type='full', random_state=51).fit(Xs)
                preds = model.predict(Xs)
                db = float(skm.davies_bouldin_score(Xs, preds)) if Xs.shape[0] > ui else float('inf')
            except Exception:
                db = float('inf')
                preds = None
            DB.append(db)
            if preds is not None and db < min(DB):
                best_preds = preds
        try:
            best_k = candidates[int(np.nanargmin(np.array(DB)))]
            # refit to get preds
            model = GaussianMixture(n_components=best_k, covariance_type='full', random_state=51).fit(Xs)
            preds = model.predict(Xs)
        except Exception:
            preds = np.zeros((Xs.shape[0],), dtype=int)
            best_k = k_min
        result['params'] = {'k': int(best_k)}
        result['stage2'] = compute_stage2_from_preds(test_X, test_Y, label_hat, theta, preds, num_known, stage2_dominance, force_clusterize=force_clusterize)
    elif method == 'minibatchkmeans':
        import sklearn.metrics as skm
        DB = []
        candidates = list(range(k_min, k_max+1))
        for ui in candidates:
            try:
                Cluster = MiniBatchKMeans(n_clusters=ui, random_state=51).fit(Xs)
                pre_label = Cluster.labels_
                db = float(skm.davies_bouldin_score(Xs, pre_label)) if Xs.shape[0] > ui else float('inf')
            except Exception:
                db = float('inf')
            DB.append(db)
        try:
            best_k = candidates[int(np.nanargmin(np.array(DB)))]
        except Exception:
            best_k = k_min
        Cluster = MiniBatchKMeans(n_clusters=best_k, random_state=51).fit(Xs)
        preds = Cluster.labels_
        result['params'] = {'k': int(best_k)}
        result['stage2'] = compute_stage2_from_preds(test_X, test_Y, label_hat, theta, preds, num_known, stage2_dominance, force_clusterize=force_clusterize)
    elif method == 'dbscan':
        best = None
        best_db = float('inf')
        best_eps = None
        import sklearn.metrics as skm
        for eps in dbscan_eps_list:
            try:
                Cluster = DBSCAN(eps=eps, min_samples=5).fit(Xs)
                preds = Cluster.labels_
                # compute DB only if there are >=2 clusters
                ncl = len(set(preds)) - (1 if -1 in preds else 0)
                if ncl >= 2:
                    db = float(skm.davies_bouldin_score(Xs, preds))
                else:
                    db = float('inf')
            except Exception:
                db = float('inf')
                preds = None
            if preds is not None and db < best_db:
                best_db = db
                best = (eps, preds)
        if best is None:
            preds = np.full((Xs.shape[0],), -1, dtype=int)
            best_eps = dbscan_eps_list[0]
        else:
            best_eps, preds = best
        result['params'] = {'eps': float(best_eps)}
        result['stage2'] = compute_stage2_from_preds(test_X, test_Y, label_hat, theta, preds, num_known, stage2_dominance, force_clusterize=force_clusterize)
    else:
        raise ValueError('Unknown method')

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--out_json', default='stage2_open_set_metrics_by_method.json')
    parser.add_argument('--k_min', type=int, default=2)
    parser.add_argument('--k_max', type=int, default=14)
    parser.add_argument('--dbscan_eps', type=str, default='0.5,1.0,1.5,2.0')
    parser.add_argument('--force_clusterize', action='store_true', help='If set, skip the u=1 compactness check and force clustering evaluation')
    parser.add_argument('--stage2_dominance', type=float, default=0.4)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    methods = ['kmeans', 'agglomerative', 'gmm', 'minibatchkmeans', 'dbscan']
    eps_list = [float(x) for x in args.dbscan_eps.split(',') if x.strip()]

    out = {'exp_dir': exp_dir, 'results': {}}
    for m in methods:
        print(f'Running {m}...')
        try:
            r = cluster_and_eval(exp_dir, m, k_min=args.k_min, k_max=args.k_max, dbscan_eps_list=eps_list, stage2_dominance=args.stage2_dominance, force_clusterize=args.force_clusterize)
            out['results'][m] = r
        except Exception as e:
            out['results'][m] = {'error': str(e)}

    out_path = os.path.join(exp_dir, args.out_json)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2)
    print('Saved metrics JSON to', out_path)

    # print concise table
    def fmt(x):
        try:
            if x is None:
                return 'nan'
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    print('\nSummary table (method | u | UP | mean_unknown_acc):')
    for m in methods:
        r = out['results'].get(m, {})
        st2 = r.get('stage2') if isinstance(r, dict) else None
        if st2 and 'UP' in st2:
            u = st2.get('u')
            up = fmt(st2.get('UP'))
            mean_acc = fmt(st2.get('mean_unknown_acc'))
            print(f"{m:12s} | u={str(u):3} | UP={up} | mean_unknown_acc={mean_acc}")
        else:
            print(f"{m:12s} | ERROR or no result")

if __name__ == '__main__':
    main()
