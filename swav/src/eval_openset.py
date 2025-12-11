import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from logging import getLogger
from matplotlib.lines import Line2D
import os
import sys
import importlib

# Optional sklearn imports for stage2 clustering
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    import sklearn.metrics as skm
except Exception:
    MinMaxScaler = None
    KMeans = None
    skm = None

logger = getLogger()

def outlier_check(distance_list):
    distance = np.flip(np.sort(distance_list))
    distance_std = np.std(np.hstack([distance, -distance]))
    threshold = distance[0]
    for index in range(distance.shape[0]):
        threshold = distance[index]
        if distance[index] <= 3 * distance_std:
            break
    return threshold

def metrics_stage_1(true_label, predict_label, num_known):
    num_samples = predict_label.shape[0]
    ones = np.ones(num_samples)
    # TKR:
    #  A: the number of known samples are accepted / B: the number of known samples
    a = np.sum(np.logical_and(predict_label != (-ones), true_label != (-ones)))
    b = np.sum(true_label != (-ones))
    tkr = a / b if b != 0 else 0

    # TUR:
    #  A: the number of unknown are rejected / B: the number of unknown samples
    a = np.sum(np.logical_and(true_label == (-ones), predict_label == (-ones)))
    b = np.sum(true_label == (-ones))
    tur = a / b if b != 0 else 0

    # KP:
    # A: the number of known samples are accurately classified / the number of all accepted samples
    a = np.sum(true_label[true_label != (-ones)] == predict_label[true_label != (-ones)])
    b = np.sum(predict_label != (-ones))
    if (b == 0):
        kp = 0
    else:
        kp = a / b

    # FKR:
    # A: the number of unknown samples are accurately rejected / the number of all rejected samples
    a = np.sum(true_label[true_label == (-ones)] == predict_label[true_label == (-ones)])
    b = np.sum(predict_label == (-ones))
    fkr = a / b if b != 0 else 0

    # Known Accuracy
    accuracy = []
    for ii in range(num_known):
        a = np.sum(np.logical_and(true_label == (ii * ones), predict_label == (ii * ones)))
        b = np.sum(true_label == (ii * ones))
        if b != 0:
            acc = a / b
            accuracy.append(acc)
        else:
            accuracy.append(-1)
    return tkr, tur, kp, fkr, accuracy

def compute_distances(train_X, train_Y, test_X, num_known, metric='mahalanobis'):
    semantic_dim = train_X.shape[1]
    theta = torch.zeros(num_known)
    
    # Store centers and precision matrices (or Identity for Euclidean)
    class_centers = torch.zeros((num_known, semantic_dim))
    dist_matrices = np.zeros((num_known, semantic_dim, semantic_dim))
    
    # 1. Fit (Calculate centers and thresholds)
    for clas in range(num_known):
        mask = train_Y == clas
        if mask.sum() == 0:
            continue
        samples = train_X[mask].numpy()
        class_centers[clas] = torch.mean(train_X[mask], dim=0)
        
        if metric == 'mahalanobis':
            covariance_mat = np.cov(samples, rowvar=False, bias=True)
            dist_matrices[clas] = np.linalg.pinv(covariance_mat)
        elif metric == 'euclidean':
            dist_matrices[clas] = np.eye(semantic_dim)
            
        x = (train_X[mask] - class_centers[clas].expand([samples.shape[0], semantic_dim]))
        x = x.numpy()
        # d = sqrt(x^T M x)
        # Diagonal of x M x^T
        temp = np.matmul(x, dist_matrices[clas])
        dist_list = np.sqrt(np.maximum(np.sum(temp * x, axis=1), 0))
        theta[clas] = outlier_check(dist_list)

    # 2. Predict (Calculate distances for test data)
    d_ct = np.zeros((test_X.shape[0], num_known))
    for xi in range(num_known):
        x = (test_X - class_centers[xi]).numpy()
        temp = np.matmul(x, dist_matrices[xi])
        d_ct[:, xi] = np.sqrt(np.maximum(np.sum(temp * x, axis=1), 0))
        
    return d_ct, theta

def evaluate_metric(test_Y, d_ct, theta, num_known, metric_name):
    Theta = theta.expand([test_Y.shape[0], num_known]).numpy()
    x_ct = d_ct - Theta
    label_hat = np.zeros(test_Y.shape[0])

    for xi in range(test_Y.shape[0]):
        if np.min(x_ct[xi]) > 0:
            label_hat[xi] = -1  # Unknown
        else:
            label_hat[xi] = np.argmin(x_ct[xi])

    test_Y_normalized = test_Y.numpy().copy()
    test_Y_normalized[test_Y_normalized >= num_known] = -1

    tkr, tur, kp, fkr, accuracy = metrics_stage_1(test_Y_normalized, label_hat, num_known)

    logger.info(f"--- {metric_name} Results ---")
    logger.info(f"TKR: {tkr:.4f}, TUR: {tur:.4f}, KP: {kp:.4f}, FKR: {fkr:.4f}")
    logger.info(f"Mean Known Accuracy: {np.mean(accuracy):.4f}")
    # Return label_hat so callers can compute additional metrics (e.g. UP / predicted-unknown precision)
    return tkr, tur, kp, fkr, np.mean(accuracy), label_hat

def plot_tsne(test_X, test_Y, num_known, dump_path):
    logger.info("Generating t-SNE plots...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(test_X.numpy())
    
    test_Y_np = test_Y.numpy()
    known_mask = test_Y_np < num_known
    unknown_mask = test_Y_np >= num_known
    
    # Helper for plotting
    def plot_scatter(x, y, labels, title, filename, cmap='tab20', alpha=0.7, show_legend=True):
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(x, y, c=labels, cmap=cmap, s=20, alpha=alpha)
        if show_legend:
            # Replace colorbar with a legend constructed from unique labels and their colors
            try:
                unique_labels = np.unique(labels)
                cmap_obj = plt.get_cmap(cmap)
                handles = []
                if unique_labels.shape[0] == 1:
                    # Single label -> single legend entry
                    color = cmap_obj(0.5)
                    handles.append(Line2D([0], [0], marker='o', color='w', label=str(unique_labels[0]),
                                          markerfacecolor=color, markersize=8))
                else:
                    # Map each unique label to a distinct color from the colormap
                    n = unique_labels.shape[0]
                    for i, lab in enumerate(unique_labels):
                        # normalize index to [0,1]
                        idx = 0 if n == 1 else float(i) / (n - 1)
                        color = cmap_obj(idx)
                        handles.append(Line2D([0], [0], marker='o', color='w', label=str(lab),
                                              markerfacecolor=color, markersize=6))
                plt.legend(handles=handles, fontsize=10, title='Class ID')
            except Exception:
                # Fallback to colorbar if legend construction fails
                plt.colorbar(scatter, label='Class ID')
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(dump_path, filename), dpi=300)
        plt.close()

    # 1. Known
    if known_mask.sum() > 0:
        plot_scatter(
            X_embedded[known_mask, 0], 
            X_embedded[known_mask, 1], 
            test_Y_np[known_mask], 
            "t-SNE: Known Classes", 
            "tsne_known.png"
        )

    # 2. Unknown
    if unknown_mask.sum() > 0:
        plot_scatter(
            X_embedded[unknown_mask, 0], 
            X_embedded[unknown_mask, 1], 
            test_Y_np[unknown_mask], 
            "t-SNE: Unknown Classes", 
            "tsne_unknown.png"
        )

    # 3. Both: plot known and unknown together.
    plt.figure(figsize=(12, 10))

    # Plot knowns (colored by class) but do NOT create a colorbar legend
    if known_mask.sum() > 0:
        plt.scatter(
            X_embedded[known_mask, 0],
            X_embedded[known_mask, 1],
            c=test_Y_np[known_mask],
            cmap='tab20',
            s=20,
            alpha=0.8,
            linewidths=0,
        )

    # Plot unknowns: all unknown classes use a single color and a thin black edge
    if unknown_mask.sum() > 0:
        unknown_color = 'tab:red'
        plt.scatter(
            X_embedded[unknown_mask, 0],
            X_embedded[unknown_mask, 1],
            color=unknown_color,
            s=36,
            alpha=0.95,
            edgecolors='k',
            linewidths=0.25,
            label='Unknown'
        )

    # Build a simple legend with two entries: Known and Unknown
    legend_handles = []
    try:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label='Known',
                                     markerfacecolor='tab:blue', markersize=8))
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label='Unknown',
                                     markerfacecolor=unknown_color, markeredgecolor='k', markersize=8))
        plt.legend(handles=legend_handles, fontsize=12)
    except Exception:
        plt.legend(fontsize=12)

    plt.title("t-SNE: Known vs Unknown", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(dump_path, "tsne_both.png"), dpi=300)
    plt.close()

def plot_distance_histogram(d_ct, test_Y, num_known, metric_name, dump_path):
    logger.info(f"Generating histogram for {metric_name}...")
    # Min distance to any known center
    min_dists = np.min(d_ct, axis=1)
    
    test_Y_np = test_Y.numpy()
    known_mask = test_Y_np < num_known
    unknown_mask = test_Y_np >= num_known
    
    plt.figure(figsize=(10, 6))
    
    # Determine bins to ensure alignment across both distributions
    if len(min_dists) > 0:
        data_min = min_dists.min()
        data_max = min_dists.max()
        bins = np.linspace(data_min, data_max, 50)
    else:
        bins = 50

    # Plot Known
    if np.sum(known_mask) > 0:
        data_known = min_dists[known_mask]
        plt.hist(data_known, bins=bins, alpha=0.5, label='Known', color='tab:blue', density=True, edgecolor='blue')
        # Add mean line
        plt.axvline(data_known.mean(), color='blue', linestyle='dashed', linewidth=1.5, label=f'Known Mean ({data_known.mean():.2f})')
        
    # Plot Unknown
    if np.sum(unknown_mask) > 0:
        data_unknown = min_dists[unknown_mask]
        plt.hist(data_unknown, bins=bins, alpha=0.5, label='Unknown', color='tab:red', density=True, edgecolor='red')
        # Add mean line
        plt.axvline(data_unknown.mean(), color='red', linestyle='dashed', linewidth=1.5, label=f'Unknown Mean ({data_unknown.mean():.2f})')
        
    plt.title(f"Distribution of {metric_name}\n(Min Distance to Known Centers)", fontsize=14)
    plt.xlabel("Distance", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    filename = f"hist_{metric_name.lower().replace(' ', '_')}.png"
    plt.savefig(os.path.join(dump_path, filename), dpi=300)
    plt.close()

def evaluate_openset(model, train_loader, test_loader, unknown_loader, args):
    logger.info("Starting Open Set Evaluation...")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Extract features for known train data to calculate centers
    logger.info("Extracting features from training data...")
    train_features = []
    train_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(train_loader):
            if isinstance(inputs, list):
                inputs = inputs[0] # Use the first crop for evaluation
            inputs = inputs.to(device)
            
            # Forward pass
            ret = model(inputs)
            if len(ret) == 3:
                embedding, output, logits = ret
            else:
                embedding, output = ret
            
            feats = output.cpu()
            train_features.append(feats)
            train_labels.append(labels)
            
    train_X = torch.cat(train_features, dim=0)
    train_Y = torch.cat(train_labels, dim=0)
    
    num_known = args.num_classes
    
    # 2. Extract features for test data (known + unknown)
    logger.info("Extracting features from test data...")
    test_features = []
    test_labels = []
    
    # Known test data
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            ret = model(inputs)
            if len(ret) == 3:
                embedding, output, logits = ret
            else:
                embedding, output = ret
            test_features.append(output.cpu())
            test_labels.append(labels)
            
    # Unknown test data
    if unknown_loader:
        with torch.no_grad():
            for inputs, labels in tqdm(unknown_loader):
                inputs = inputs.to(device)
                ret = model(inputs)
                if len(ret) == 3:
                    embedding, output, logits = ret
                else:
                    embedding, output = ret
                test_features.append(output.cpu())
                test_labels.append(labels)
                
    test_X = torch.cat(test_features, dim=0)
    test_Y = torch.cat(test_labels, dim=0)
    
    # 3. Evaluate Euclidean Distance
    logger.info("Evaluating with Euclidean Distance...")
    d_ct_eu, theta_eu = compute_distances(train_X, train_Y, test_X, num_known, metric='euclidean')
    tkr_eu, tur_eu, kp_eu, fkr_eu, mean_acc_eu, label_hat_eu = evaluate_metric(test_Y, d_ct_eu, theta_eu, num_known, "Euclidean Distance")

    # Compute UP (predicted-unknown precision) for Euclidean evaluation:
    try:
        test_Y_np = test_Y.numpy().copy()
        test_Y_np[test_Y_np >= num_known] = -1
        pred_unknown_mask = (label_hat_eu == -1)
        c = int(np.sum(pred_unknown_mask))
        a = int(np.sum(test_Y_np[pred_unknown_mask] == -1)) if c > 0 else 0
        UP_eu = float(a / c) if c > 0 else float('nan')
        logger.info(f"UP (Euclidean): {UP_eu:.4f} (predicted_unknowns={c}, correct_unknowns={a})")
    except Exception:
        logger.info("UP (Euclidean): could not be computed")

    # --- Stage2-style clustering-based UP (u>1) for Euclidean ---
    try:
        # Try to import the helper from scripts if available
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        stage2_mod = None
        try:
            stage2_mod = importlib.import_module('scripts.stage2_compute_open_set_metrics')
        except Exception:
            stage2_mod = None

        # Prepare arrays
        test_X_np = test_X.numpy()
        test_Y_np = test_Y.numpy()
        label_hat_np = label_hat_eu

        unknown_idx = np.where(label_hat_np == -1)[0]
        if unknown_idx.size == 0:
            logger.info('Stage2 (Euclidean): no unknown samples predicted by stage1; skipping clustering.')
        else:
            X_unknown = test_X_np[unknown_idx]
            y_unknown = test_Y_np[unknown_idx]

            # If stage2 helper available, prefer using its compute_stage2_from_preds after KMeans clustering
            if stage2_mod is not None and MinMaxScaler is not None and KMeans is not None:
                scaler = MinMaxScaler()
                Xs = scaler.fit_transform(X_unknown) if X_unknown.shape[0] > 0 else X_unknown
                # Search k by DB index (ensure candidates <= n_samples and inclusive)
                k_min = 2
                k_max = 14
                n_samples = Xs.shape[0]
                max_k = min(k_max, n_samples)
                if max_k < k_min:
                    candidates = [k_min]
                else:
                    candidates = list(range(k_min, max_k + 1))

                DB = []
                db_map = {}
                for ui in candidates:
                    try:
                        Cluster = KMeans(n_clusters=ui, init='k-means++', random_state=51).fit(Xs)
                        pre_label = Cluster.labels_
                        # Davies-Bouldin requires at least 2 clusters; otherwise inf
                        db = float(skm.davies_bouldin_score(Xs, pre_label)) if Xs.shape[0] > ui and len(set(pre_label)) > 1 else float('inf')
                    except Exception:
                        db = float('inf')
                    DB.append(db)
                    db_map[ui] = db

                # Log DB scores to help debugging
                try:
                    logger.info(f"Stage2 (Euclidean clustering): DB scores by k: {db_map}")
                except Exception:
                    pass

                try:
                    best_k = candidates[int(np.nanargmin(np.array(DB)))]
                except Exception:
                    best_k = candidates[0]

                Cluster = KMeans(n_clusters=best_k, init='k-means++', random_state=51).fit(Xs)
                preds = Cluster.labels_

                # call compute_stage2_from_preds from module and log detailed outputs
                try:
                    st2 = stage2_mod.compute_stage2_from_preds(test_X_np, test_Y_np, label_hat_np, theta_eu.numpy() if hasattr(theta_eu, 'numpy') else theta_eu, preds, num_known)
                    if isinstance(st2, dict):
                        logger.info(f"Stage2 (Euclidean clustering) chosen_k={best_k} -> {st2}")
                    else:
                        logger.info('Stage2 (Euclidean clustering): result has no UP')
                except Exception as e:
                    logger.info(f'Stage2 (Euclidean clustering): failed to compute stage2 via helper: {e}')
            else:
                logger.info('Stage2 (Euclidean clustering): sklearn or helper not available; skipping clustering-based UP')
    except Exception:
        logger.info('Stage2 (Euclidean clustering): unexpected error; skipping')
    
    # 4. Evaluate Mahalanobis Distance
    logger.info("Evaluating with Mahalanobis Distance...")
    d_ct_ma, theta_ma = compute_distances(train_X, train_Y, test_X, num_known, metric='mahalanobis')
    tkr_ma, tur_ma, kp_ma, fkr_ma, mean_acc_ma, label_hat_ma = evaluate_metric(test_Y, d_ct_ma, theta_ma, num_known, "Mahalanobis Distance")

    # Compute UP (predicted-unknown precision) for Mahalanobis evaluation:
    try:
        test_Y_np = test_Y.numpy().copy()
        test_Y_np[test_Y_np >= num_known] = -1
        pred_unknown_mask = (label_hat_ma == -1)
        c = int(np.sum(pred_unknown_mask))
        a = int(np.sum(test_Y_np[pred_unknown_mask] == -1)) if c > 0 else 0
        UP_ma = float(a / c) if c > 0 else float('nan')
        logger.info(f"UP (Mahalanobis): {UP_ma:.4f} (predicted_unknowns={c}, correct_unknowns={a})")
    except Exception:
        logger.info("UP (Mahalanobis): could not be computed")

    # --- Stage2-style clustering-based UP (u>1) for Mahalanobis ---
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        try:
            stage2_mod = importlib.import_module('scripts.stage2_compute_open_set_metrics')
        except Exception:
            stage2_mod = None

        test_X_np = test_X.numpy()
        test_Y_np = test_Y.numpy()
        label_hat_np = label_hat_ma

        unknown_idx = np.where(label_hat_np == -1)[0]
        if unknown_idx.size == 0:
            logger.info('Stage2 (Mahalanobis): no unknown samples predicted by stage1; skipping clustering.')
        else:
            X_unknown = test_X_np[unknown_idx]
            y_unknown = test_Y_np[unknown_idx]

            if stage2_mod is not None and MinMaxScaler is not None and KMeans is not None:
                scaler = MinMaxScaler()
                Xs = scaler.fit_transform(X_unknown) if X_unknown.shape[0] > 0 else X_unknown
                # build candidate k list inclusive and <= n_samples
                k_min = 2
                k_max = 14
                n_samples = Xs.shape[0]
                max_k = min(k_max, n_samples)
                if max_k < k_min:
                    candidates = [k_min]
                else:
                    candidates = list(range(k_min, max_k + 1))

                DB = []
                db_map = {}
                for ui in candidates:
                    try:
                        Cluster = KMeans(n_clusters=ui, init='k-means++', random_state=51).fit(Xs)
                        pre_label = Cluster.labels_
                        db = float(skm.davies_bouldin_score(Xs, pre_label)) if Xs.shape[0] > ui and len(set(pre_label)) > 1 else float('inf')
                    except Exception:
                        db = float('inf')
                    DB.append(db)
                    db_map[ui] = db

                try:
                    logger.info(f"Stage2 (Mahalanobis clustering): DB scores by k: {db_map}")
                except Exception:
                    pass

                try:
                    best_k = candidates[int(np.nanargmin(np.array(DB)))]
                except Exception:
                    best_k = candidates[0]

                Cluster = KMeans(n_clusters=best_k, init='k-means++', random_state=51).fit(Xs)
                preds = Cluster.labels_
                try:
                    st2 = stage2_mod.compute_stage2_from_preds(test_X_np, test_Y_np, label_hat_np, theta_ma.numpy() if hasattr(theta_ma, 'numpy') else theta_ma, preds, num_known)
                    if isinstance(st2, dict):
                        logger.info(f"Stage2 (Mahalanobis clustering) chosen_k={best_k} -> {st2}")
                    else:
                        logger.info('Stage2 (Mahalanobis clustering): result has no UP')
                except Exception as e:
                    logger.info(f'Stage2 (Mahalanobis clustering): failed to compute stage2 via helper: {e}')
            else:
                logger.info('Stage2 (Mahalanobis clustering): sklearn or helper not available; skipping clustering-based UP')
    except Exception:
        logger.info('Stage2 (Mahalanobis clustering): unexpected error; skipping')
    
    # 5. t-SNE Plotting
    plot_tsne(test_X, test_Y, num_known, args.dump_path)
    
    # 6. Distance Histogram
    plot_distance_histogram(d_ct_eu, test_Y, num_known, "Euclidean Distance", args.dump_path)
    plot_distance_histogram(d_ct_ma, test_Y, num_known, "Mahalanobis Distance", args.dump_path)
    
    logger.info("Evaluation complete. Plots saved to dump path.")
