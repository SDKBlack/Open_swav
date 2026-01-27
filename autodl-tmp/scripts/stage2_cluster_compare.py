"""
Compare multiple clustering algorithms on the Stage-2 unknown features saved by `main_swav.py`.
Loads: test_X.npy, test_Y.npy, label_hat.npy from an experiment directory.
Saves results JSON to <exp_dir>/stage2_cluster_compare_results.json and prints a concise table.

Usage: python3 scripts/stage2_cluster_compare.py --exp_dir /path/to/exp_dir
"""
import os
import argparse
import json
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score


def evaluate_clustering(X_true, y_true, labels_pred):
    res = {}
    n_clusters = len(set(labels_pred)) - (1 if -1 in labels_pred else 0)
    res['n_clusters'] = int(n_clusters)
    try:
        if n_clusters >= 2 and X_true.shape[0] >= 2:
            res['silhouette'] = float(silhouette_score(X_true, labels_pred))
        else:
            res['silhouette'] = None
    except Exception:
        res['silhouette'] = None
    try:
        if n_clusters >= 2:
            res['davies_bouldin'] = float(davies_bouldin_score(X_true, labels_pred))
        else:
            res['davies_bouldin'] = None
    except Exception:
        res['davies_bouldin'] = None
    try:
        if n_clusters >= 2:
            res['calinski_harabasz'] = float(calinski_harabasz_score(X_true, labels_pred))
        else:
            res['calinski_harabasz'] = None
    except Exception:
        res['calinski_harabasz'] = None
    try:
        # If ground truth labels for unknown set are provided, compute ARI
        if y_true is not None:
            res['adjusted_rand_index'] = float(adjusted_rand_score(y_true, labels_pred))
        else:
            res['adjusted_rand_index'] = None
    except Exception:
        res['adjusted_rand_index'] = None
    return res


def run_k_search(X, y_true, k_min, k_max, method_name='kmeans'):
    best_db = 1e12
    best_s = None
    best_info = None
    for k in range(k_min, k_max + 1):
        try:
            if method_name == 'kmeans':
                model = KMeans(n_clusters=k, init='k-means++', random_state=51)
            elif method_name == 'mbkmeans':
                model = MiniBatchKMeans(n_clusters=k, random_state=51)
            elif method_name == 'gmm':
                model = GaussianMixture(n_components=k, covariance_type='full', random_state=51)
            elif method_name == 'agg':
                # ward requires Euclidean distance and dense features
                model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            else:
                continue
            if method_name == 'gmm':
                preds = model.fit_predict(X)
            else:
                preds = model.fit(X).predict(X) if hasattr(model, 'predict') else model.fit_predict(X)
            info = evaluate_clustering(X, y_true, preds)
            info['k'] = int(k)
            # choose by Davies-Bouldin when available, else silhouette (higher better)
            score_db = info.get('davies_bouldin')
            if score_db is None:
                score_db = 1e6 if info.get('silhouette') is None else -info['silhouette']
            if score_db < best_db:
                best_db = score_db
                best_info = dict(method=method_name, preds=preds.tolist(), metrics=info)
        except Exception as e:
            # skip failing k
            # print(f"skipping {method_name} k={k} due to {e}")
            continue
    return best_info


def run_dbscan_grid(X, y_true, eps_list, min_samples=5):
    results = []
    for eps in eps_list:
        try:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            preds = model.fit_predict(X)
            info = evaluate_clustering(X, y_true, preds)
            info['eps'] = float(eps)
            info['n_noise'] = int((preds == -1).sum())
            results.append({'method': 'dbscan', 'preds': preds.tolist(), 'metrics': info})
        except Exception:
            continue
    # pick best by davies_bouldin if available
    best = None
    best_db = 1e12
    for r in results:
        db = r['metrics'].get('davies_bouldin')
        if db is None:
            continue
        if db < best_db:
            best_db = db
            best = r
    return best, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True, help='experiment directory with test_X.npy etc')
    parser.add_argument('--k_min', type=int, default=2)
    parser.add_argument('--k_max', type=int, default=14)
    parser.add_argument('--dbscan_eps', type=str, default='0.5,1.0,1.5,2.0')
    parser.add_argument('--min_samples', type=int, default=5)
    parser.add_argument('--out_json', type=str, default='stage2_cluster_compare_results.json')
    args = parser.parse_args()

    p = args.exp_dir
    if not os.path.isdir(p):
        raise RuntimeError(f'exp_dir not found: {p}')
    test_X_path = os.path.join(p, 'test_X.npy')
    label_hat_path = os.path.join(p, 'label_hat.npy')
    test_Y_path = os.path.join(p, 'test_Y.npy')
    if not os.path.exists(test_X_path):
        raise RuntimeError('test_X.npy not found in exp_dir')
    X = np.load(test_X_path)
    label_hat = np.load(label_hat_path) if os.path.exists(label_hat_path) else None
    test_Y = np.load(test_Y_path) if os.path.exists(test_Y_path) else None

    # select unknown subset where label_hat == -1
    if label_hat is None:
        raise RuntimeError('label_hat.npy required (to select unknown samples)')
    unk_mask = (label_hat == -1)
    X_unk = X[unk_mask]
    y_unk = test_Y[unk_mask] if test_Y is not None else None

    results = {'exp_dir': p, 'n_unknown': int(X_unk.shape[0])}

    # KMeans baseline (repeat the search as main_swav.py)
    kmeans_best = run_k_search(X_unk, y_unk, args.k_min, args.k_max, method_name='kmeans')
    results['kmeans_best'] = kmeans_best['metrics'] if kmeans_best else None

    # Agglomerative (search k)
    agg_best = run_k_search(X_unk, y_unk, args.k_min, args.k_max, method_name='agg')
    results['agglomerative_best'] = agg_best['metrics'] if agg_best else None

    # Gaussian Mixture (search k)
    gmm_best = run_k_search(X_unk, y_unk, args.k_min, args.k_max, method_name='gmm')
    results['gmm_best'] = gmm_best['metrics'] if gmm_best else None

    # MiniBatchKMeans
    mbk_best = run_k_search(X_unk, y_unk, args.k_min, args.k_max, method_name='mbkmeans')
    results['minibatchkmeans_best'] = mbk_best['metrics'] if mbk_best else None

    # DBSCAN grid
    eps_list = [float(x) for x in args.dbscan_eps.split(',') if x.strip()]
    db_best, db_all = run_dbscan_grid(X_unk, y_unk, eps_list, min_samples=args.min_samples)
    results['dbscan_best'] = db_best['metrics'] if db_best else None
    results['dbscan_all'] = [r['metrics'] for r in db_all]

    # Save JSON
    out_path = os.path.join(p, args.out_json)
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=2)
    print('Saved results to', out_path)

    # Print concise table
    def short(m):
        if m is None:
            return 'None'
        return f"k={m.get('k', '-')}, clusters={m.get('n_clusters')}, ARI={m.get('adjusted_rand_index'):.3f} DB={m.get('davies_bouldin'):.3f} Sil={m.get('silhouette'):.3f}"
    print('\nSummary:')
    print('KMeans :', short(results['kmeans_best']))
    print('Agglo  :', short(results['agglomerative_best']))
    print('GMM    :', short(results['gmm_best']))
    print('MB-KMeans:', short(results['minibatchkmeans_best']))
    print('DBSCAN best:', short(results['dbscan_best']))


if __name__ == '__main__':
    main()
