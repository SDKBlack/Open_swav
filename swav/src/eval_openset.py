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
    from sklearn import metrics
except Exception:
    MinMaxScaler = None
    KMeans = None
    skm = None
    metrics = None

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

def compute_stage2_up(test_X, test_Y, label_hat, theta, num_known):
    if metrics is None or KMeans is None or MinMaxScaler is None:
        logger.info("Stage 2 UP: Sklearn components not available.")
        return {'up_db': 0.0, 'up_sil': 0.0, 'up_kmeans': 0.0}

    # Prepare data
    test_X_np = test_X.numpy() if isinstance(test_X, torch.Tensor) else test_X
    test_Y_np = test_Y.numpy() if isinstance(test_Y, torch.Tensor) else test_Y
    
    # Normalize test_Y for unknown handling (unknowns >= num_known -> -1)
    test_Y_normalized = test_Y_np.copy()
    test_Y_normalized[test_Y_normalized >= num_known] = -1
    
    # Filter predicted unknowns
    unknown_mask = (label_hat == -1)
    if np.sum(unknown_mask) == 0:
        logger.info("Stage 2 UP: No samples predicted as unknown.")
        return {
            'up_db': 0.0,
            'up_sil': 0.0,
            'up_kmeans': 0.0,
            'unknown_acc_db': {},
            'unknown_acc_sil': {},
            'unknown_acc_kmeans': {},
            'unknown_acc_mean_db': 0.0,
            'unknown_acc_mean_sil': 0.0,
            'unknown_acc_mean_kmeans': 0.0,
        }

    # Use the passed test_X (which is now test_Combined) for clustering
    predict_unknown_X = test_X_np[unknown_mask]
    predict_unknown_Y = test_Y_np[unknown_mask]
    
    # Determine num_unknown from the data (assuming test_Y contains all classes)
    max_label = np.max(test_Y_np)
    num_unknown = max(0, max_label - num_known + 1)
    
    # --- u=1 Check ---
    # Calculate theta for all predict_unknown_X
    try:
        covariance_mat = np.cov(predict_unknown_X, rowvar=False, bias=True)
        matrix = np.linalg.pinv(covariance_mat)
        centers = np.mean(predict_unknown_X, axis=0)
        x = (predict_unknown_X - centers)
        # Mahalanobis distance
        dist_list = np.sqrt(np.sum(np.matmul(x, matrix) * x, axis=1))
        theta_u1 = outlier_check(dist_list)
        
        theta_max = np.max(theta) if isinstance(theta, np.ndarray) else theta.max().item()
        
        if theta_u1 <= theta_max:
            logger.info("Stage 2 UP: u=1 detected (Compact).")
            # Calculate UP (Precision of Unknowns)
            # a: number of true unknowns in predicted unknowns
            a = np.sum(test_Y_normalized[unknown_mask] == -1)
            c = predict_unknown_X.shape[0]
            up = a / c if c > 0 else 0.0
            # With u=1 we cannot classify per unknown class (only one cluster).
            # Still return empty per-class accuracies for API consistency.
            return {
                'up_db': up,
                'up_sil': up,
                'up_kmeans': up,
                'unknown_acc_db': {},
                'unknown_acc_sil': {},
                'unknown_acc_kmeans': {},
                'unknown_acc_mean_db': 0.0,
                'unknown_acc_mean_sil': 0.0,
                'unknown_acc_mean_kmeans': 0.0,
            }
    except Exception as e:
        logger.info(f"Stage 2 UP: u=1 check failed ({e}), proceeding to clustering.")

    # --- Clustering (u > 1) ---
    try:
        scaler = MinMaxScaler()
        predict_unknown_X_scaled = scaler.fit_transform(predict_unknown_X)
        
        db_scores = []
        sil_scores = []
        
        # Search for best k
        k_min = 2
        k_max = 15
        n_samples = predict_unknown_X_scaled.shape[0]
        candidates = range(k_min, min(k_max, n_samples))
        
        if len(candidates) == 0:
             # Fallback if too few samples
             candidates = [min(2, n_samples)] if n_samples > 0 else []

        for ui in candidates:
            if ui < 2: continue
            kmeans = KMeans(n_clusters=ui, init='k-means++', random_state=51).fit(predict_unknown_X_scaled)
            pre_label = kmeans.labels_
            if len(np.unique(pre_label)) > 1:
                # Davies-Bouldin (Lower is better)
                db = metrics.davies_bouldin_score(predict_unknown_X_scaled, pre_label)
                # Silhouette Score (Higher is better)
                sil = metrics.silhouette_score(predict_unknown_X_scaled, pre_label)
            else:
                db = float('inf')
                sil = -1.0
            db_scores.append(db)
            sil_scores.append(sil)
            
        if not db_scores:
            logger.info("Stage 2 UP: Clustering failed (not enough samples/clusters).")
            return {
                'up_db': 0.0,
                'up_sil': 0.0,
                'up_kmeans': 0.0,
                'unknown_acc_db': {},
                'unknown_acc_sil': {},
                'unknown_acc_kmeans': {},
                'unknown_acc_mean_db': 0.0,
                'unknown_acc_mean_sil': 0.0,
                'unknown_acc_mean_kmeans': 0.0,
            }

        # Select k
        u_db = candidates[np.argmin(db_scores)]
        u_sil = candidates[np.argmax(sil_scores)]
        
        logger.info(f"Stage 2 UP: Selected u_db={u_db} (DB Score), u_sil={u_sil} (Silhouette Score)")
        
        def _majority_map_and_acc(true_labels, cluster_labels, num_known_local):
            """Map each cluster -> unknown class by majority vote and compute per-unknown-class accuracy.

            true_labels: shape [N], original labels in dataset space (>= num_known for unknown)
            cluster_labels: shape [N], values in [0..K-1]
            """
            true_labels = np.asarray(true_labels)
            cluster_labels = np.asarray(cluster_labels)

            unknown_true = true_labels[true_labels >= num_known_local]
            if unknown_true.size == 0:
                return {}, 0.0

            unknown_labels_local = sorted(set(unknown_true.tolist()))
            if len(unknown_labels_local) == 0:
                return {}, 0.0

            # Build mapping cluster -> unknown_label (majority among unknown samples in that cluster)
            cluster_to_label = {}
            for c in np.unique(cluster_labels):
                idx = (cluster_labels == c)
                if np.sum(idx) == 0:
                    continue
                ys = true_labels[idx]
                ys = ys[ys >= num_known_local]
                if ys.size == 0:
                    continue
                vals, cnts = np.unique(ys, return_counts=True)
                cluster_to_label[int(c)] = int(vals[np.argmax(cnts)])

            # Predict unknown label for each sample
            pred_unknown = np.full(true_labels.shape[0], fill_value=-1, dtype=int)
            for i in range(true_labels.shape[0]):
                cl = int(cluster_labels[i])
                if cl in cluster_to_label:
                    pred_unknown[i] = int(cluster_to_label[cl])

            # Per unknown class accuracy
            per_class_acc = {}
            for u in unknown_labels_local:
                mask_u = (true_labels == u)
                tot_u = int(np.sum(mask_u))
                cor_u = int(np.sum(pred_unknown[mask_u] == u))
                per_class_acc[int(u)] = (float(cor_u) / float(tot_u)) if tot_u > 0 else 0.0

            mean_acc = float(np.mean(list(per_class_acc.values()))) if len(per_class_acc) > 0 else 0.0
            return per_class_acc, mean_acc

        def calculate_up_for_k(k_val):
            # Final clustering with selected u
            kmeans = KMeans(n_clusters=k_val, init='k-means++', random_state=51).fit(predict_unknown_X_scaled)
            pred_label = kmeans.labels_
            
            # Confusion Matrix: rows=clusters, cols=classes (known + unknown)
            total_classes = num_known + num_unknown
            confusion_mat = np.zeros((k_val, total_classes))
            
            for xi in range(predict_unknown_X.shape[0]):
                true_label = int(predict_unknown_Y[xi])
                cluster_label = int(pred_label[xi])
                if true_label < total_classes:
                    confusion_mat[cluster_label][true_label] += 1
                    
            # Slice to keep only unknown classes columns
            confusion_mat_unknown = confusion_mat[:, num_known:]
            
            dominate_sample = np.zeros(num_unknown)
            for row in range(k_val):
                for col in range(num_unknown):
                    cluster_total_unknowns = np.sum(confusion_mat_unknown[row])
                    if cluster_total_unknowns > 0:
                        if confusion_mat_unknown[row][col] >= cluster_total_unknowns * 0.5:
                            # Check if this cluster is the one with max samples for this class
                            if np.argmax(confusion_mat_unknown[:, col]) == row:
                                dominate_sample[col] = confusion_mat_unknown[row][col]
                                
            up_val = np.sum(dominate_sample) / predict_unknown_X.shape[0]

            # Unknown per-class classification accuracy (true unknown label -> predicted unknown label)
            per_unknown_acc, mean_unknown_acc = _majority_map_and_acc(predict_unknown_Y, pred_label, num_known)
            return up_val, per_unknown_acc, mean_unknown_acc

        up_db, acc_db, acc_mean_db = calculate_up_for_k(u_db)
        up_sil, acc_sil, acc_mean_sil = calculate_up_for_k(u_sil)

        # KMeans method: use k = num_unknown (fallback to 2), capped by sample count
        n_samples2 = predict_unknown_X_scaled.shape[0]
        k_kmeans = num_unknown if num_unknown >= 2 else 2
        k_kmeans = min(k_kmeans, max(2, n_samples2))
        if k_kmeans < 2 or n_samples2 < 2:
            up_kmeans, acc_kmeans, acc_mean_kmeans = 0.0, {}, 0.0
        else:
            up_kmeans, acc_kmeans, acc_mean_kmeans = calculate_up_for_k(int(k_kmeans))

        return {
            'up_db': float(up_db),
            'up_sil': float(up_sil),
            'up_kmeans': float(up_kmeans),
            'unknown_acc_db': acc_db,
            'unknown_acc_sil': acc_sil,
            'unknown_acc_kmeans': acc_kmeans,
            'unknown_acc_mean_db': float(acc_mean_db),
            'unknown_acc_mean_sil': float(acc_mean_sil),
            'unknown_acc_mean_kmeans': float(acc_mean_kmeans),
        }

    except Exception as e:
        logger.info(f"Stage 2 UP: Clustering logic failed ({e})")
        return {
            'up_db': 0.0,
            'up_sil': 0.0,
            'up_kmeans': 0.0,
            'unknown_acc_db': {},
            'unknown_acc_sil': {},
            'unknown_acc_kmeans': {},
            'unknown_acc_mean_db': 0.0,
            'unknown_acc_mean_sil': 0.0,
            'unknown_acc_mean_kmeans': 0.0,
        }

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

    # Per-class accuracy (known + unknown)
    # Known: accuracy list already contains per-class acc for [0..num_known-1]
    try:
        logger.info(f"Per-class Accuracy (Known, 0..{num_known-1}):")
        for k in range(num_known):
            ak = float(accuracy[k]) if k < len(accuracy) else -1.0
            if ak < 0:
                # class absent
                logger.info(f"  Class {k}: N/A")
            else:
                logger.info(f"  Class {k}: {ak:.4f}")
    except Exception:
        pass

    # Unknown per-class "correctly classified" accuracy is a Stage-2 concept
    # (after clustering predicted-unknown samples). We'll log it where Stage-2
    # is computed (see compute_stage2_up usage in evaluate_openset).

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
    def plot_scatter(x, y, labels, title, filename, cmap='tab20', alpha=0.7, show_legend=True, label_map=None):
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
                    lab = unique_labels[0]
                    label_text = str(lab)
                    if label_map and lab in label_map:
                        label_text = label_map[lab]
                    handles.append(Line2D([0], [0], marker='o', color='w', label=label_text,
                                          markerfacecolor=color, markersize=8))
                else:
                    # Map each unique label to a distinct color from the colormap
                    n = unique_labels.shape[0]
                    for i, lab in enumerate(unique_labels):
                        # normalize index to [0,1]
                        idx = 0 if n == 1 else float(i) / (n - 1)
                        color = cmap_obj(idx)
                        label_text = str(lab)
                        if label_map and lab in label_map:
                            label_text = label_map[lab]
                        handles.append(Line2D([0], [0], marker='o', color='w', label=label_text,
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

    # Plot all
    # Merge all unknown classes into one label for visualization
    test_Y_merged = test_Y_np.copy()
    test_Y_merged[test_Y_merged >= num_known] = num_known
    
    plot_scatter(X_embedded[:, 0], X_embedded[:, 1], test_Y_merged, "t-SNE (All Classes)", "tsne_all.png", label_map={num_known: 'Unknown'})
    
    # Plot Known only
    if np.sum(known_mask) > 0:
        plot_scatter(X_embedded[known_mask, 0], X_embedded[known_mask, 1], test_Y_np[known_mask], 
                     "t-SNE (Known Classes)", "tsne_known.png")
                     
    # Plot Unknown only
    if np.sum(unknown_mask) > 0:
        plot_scatter(X_embedded[unknown_mask, 0], X_embedded[unknown_mask, 1], test_Y_np[unknown_mask], 
                     "t-SNE (Unknown Classes)", "tsne_unknown.png")

def plot_hypersphere_projection(test_E, test_Y, num_known, dump_path, prototypes=None):
    logger.info("Generating 3D Hypersphere projection plots...")
    from sklearn.decomposition import PCA
    from mpl_toolkits.mplot3d import Axes3D

    # Normalize embeddings
    test_E_norm = torch.nn.functional.normalize(test_E, dim=1, p=2)
    
    if prototypes is not None:
        prototypes = torch.nn.functional.normalize(prototypes, dim=1, p=2)
        combined_data = torch.cat([test_E_norm, prototypes], dim=0)
        n_samples = test_E_norm.shape[0]
        n_protos = prototypes.shape[0]
    else:
        combined_data = test_E_norm
        n_samples = test_E_norm.shape[0]
        n_protos = 0
        
    # PCA to 3D
    pca = PCA(n_components=3)
    projected = pca.fit_transform(combined_data.numpy())
    
    # Re-normalize to project onto the unit sphere
    projected_norm = projected / np.linalg.norm(projected, axis=1, keepdims=True)
    
    X_proj = projected_norm[:n_samples]
    Proto_proj = projected_norm[n_samples:] if n_protos > 0 else None
    
    test_Y_np = test_Y.numpy()
    
    # Helper for plotting 3D
    def plot_3d_scatter(x, y, z, labels, title, filename, cmap='tab20', alpha=0.7, proto_coords=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw wireframe sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sx = np.cos(u)*np.sin(v)
        sy = np.sin(u)*np.sin(v)
        sz = np.cos(v)
        ax.plot_wireframe(sx, sy, sz, color="gray", alpha=0.1)
        
        scatter = ax.scatter(x, y, z, c=labels, cmap=cmap, s=20, alpha=alpha)
        
        if proto_coords is not None:
            ax.scatter(proto_coords[:, 0], proto_coords[:, 1], proto_coords[:, 2], 
                       marker='*', c='black', s=200, label='Prototypes', edgecolors='white')
            for i in range(proto_coords.shape[0]):
                ax.text(proto_coords[i, 0], proto_coords[i, 1], proto_coords[i, 2], 
                        str(i), fontsize=10, fontweight='bold', color='black')

        ax.set_title(title)
        # Hide axes
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(dump_path, filename), dpi=300)
        plt.close()

    # Plot all classes
    plot_3d_scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], test_Y_np, 
                    "3D Hypersphere Projection (All Classes)", "sphere_projection_all.png", proto_coords=Proto_proj)
    
    # Plot Known vs Unknown
    binary_labels = (test_Y_np >= num_known).astype(int)
    plot_3d_scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], binary_labels, 
                    "3D Hypersphere Projection (Known vs Unknown)", "sphere_projection_binary.png", cmap='coolwarm', proto_coords=Proto_proj)

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

def compute_osr_metrics(true_label, predict_label, num_known):
    # Helper to compute metrics given fixed labels
    # Reuses logic from metrics_stage_1 but adapted
    
    # Ensure numpy
    if isinstance(true_label, torch.Tensor): true_label = true_label.numpy()
    if isinstance(predict_label, torch.Tensor): predict_label = predict_label.numpy()
    
    # Normalize true labels for unknown handling
    true_label_norm = true_label.copy()
    true_label_norm[true_label_norm >= num_known] = -1
    
    num_samples = predict_label.shape[0]
    ones = np.ones(num_samples)
    
    # TKR: Known samples accepted / Total known samples
    # Known samples are those where true_label < num_known (or != -1 in normalized)
    known_mask = (true_label_norm != -1)
    if np.sum(known_mask) > 0:
        tkr = np.sum(predict_label[known_mask] != -1) / np.sum(known_mask)
    else:
        tkr = 0.0

    # TUR: Unknown samples rejected / Total unknown samples
    unknown_mask = (true_label_norm == -1)
    if np.sum(unknown_mask) > 0:
        tur = np.sum(predict_label[unknown_mask] == -1) / np.sum(unknown_mask)
    else:
        tur = 0.0

    # KP: Known samples accurately classified / All accepted samples
    accepted_mask = (predict_label != -1)
    if np.sum(accepted_mask) > 0:
        # Only consider samples that are truly known AND accepted
        # But KP definition is usually: Correctly Classified Knowns / All Accepted
        # Wait, standard definition:
        # Precision of known classes?
        # Let's follow metrics_stage_1 logic:
        # a = np.sum(true_label[true_label != (-ones)] == predict_label[true_label != (-ones)])
        # This line in metrics_stage_1 looks at ONLY known samples.
        # "the number of known samples are accurately classified"
        
        # Let's stick to the implementation in metrics_stage_1
        # But we need to be careful about indices.
        
        # Correctly classified knowns:
        # true_label == predict_label AND true_label is known
        correct_known = (true_label_norm != -1) & (true_label_norm == predict_label)
        
        # Denominator: "the number of all accepted samples" (predict_label != -1)
        # OR "the number of known samples" ?
        # metrics_stage_1 says: "the number of known samples are accurately classified / the number of all accepted samples"
        # But the code implementation in metrics_stage_1:
        # a = np.sum(true_label[true_label != (-ones)] == predict_label[true_label != (-ones)])
        # b = np.sum(predict_label != (-ones))
        # The 'a' part filters true_label for knowns, and checks equality.
        # But predict_label is not filtered? 
        # Actually: predict_label[true_label != -1] aligns with true_label[true_label != -1]
        # So it checks if known samples are correctly classified.
        # It does NOT penalize if an unknown sample is accepted (misclassified as known).
        # Wait, 'b' is total accepted. So if unknown is accepted, b increases, kp decreases. Correct.
        
        a = np.sum(correct_known)
        b = np.sum(accepted_mask)
        kp = a / b if b > 0 else 0.0
    else:
        kp = 0.0
        
    # FKR: Unknown samples accepted / Total unknown samples (1 - TUR)
    fkr = 1.0 - tur
    
    # Mean Known Accuracy
    # Accuracy on known classes only (ignoring rejection? or including rejection as error?)
    # Usually: Correctly classified / Total Known
    if np.sum(known_mask) > 0:
        mean_acc = np.sum(correct_known) / np.sum(known_mask)
    else:
        mean_acc = 0.0
        
    return tkr, tur, kp, fkr, mean_acc

def compute_prototype_distances(test_X, model, num_known):
    """
    使用模型自带的 Prototypes 计算余弦距离。
    Distance = 1 - CosineSimilarity
    """
    # 1. 获取 Prototypes 权重
    # 注意：根据您的代码，prototypes 可能是 nn.Linear 或 MultiPrototypes
    # 这里假设是单头 nn.Linear，如果是 MultiPrototypes 需要取 self.prototypes.prototypes0
    if hasattr(model, 'module'):
        proto_layer = model.module.prototypes
    else:
        proto_layer = model.prototypes

    # 兼容 MultiPrototypes (如果有多个头，我们通常取第一个)
    if isinstance(proto_layer, torch.nn.ModuleList) or hasattr(proto_layer, 'prototypes0'):
         # 假设 MultiPrototypes 结构
         proto_weight = getattr(proto_layer, 'prototypes0').weight.data.cpu()
    elif isinstance(proto_layer, torch.nn.Linear):
         proto_weight = proto_layer.weight.data.cpu()
    else:
        # 如果没有 prototypes (比如微调阶段去掉了)，则回退到均值中心法
        print("Warning: No prototypes found, skipping prototype distance.")
        return None

    # 2. 归一化 (SwAV 核心：在单位球面上比较)
    # test_X: [N, D], proto_weight: [K, D]
    test_X_norm = torch.nn.functional.normalize(test_X, p=2, dim=1)
    proto_norm = torch.nn.functional.normalize(proto_weight, p=2, dim=1)

    # 3. 计算余弦相似度 (Cosine Similarity)
    # Sim = X * P^T
    # Shape: [N, K]
    cosine_sim = torch.matmul(test_X_norm, proto_norm.t())

    # 4. 转换为“距离” (Distance)
    # 因为 OSR 算法通常找“最小距离”，所以用 1 - Sim
    # Sim 范围 [-1, 1], Dist 范围 [0, 2]
    d_ct = 1.0 - cosine_sim.numpy()

    # 5. 计算阈值 (Theta)
    # 对于 SwAV，已知类的样本和它对应的 Prototype 相似度应该非常高 (距离接近 0)
    # 我们依然可以用 outlier_check 来确定拒识阈值
    # 这里我们只取每个样本到"所属类" Prototype 的距离来算阈值
    # 但由于这是 Test 阶段，我们通常需要利用 Train Set 的特征来定阈值
    # 为了简化，这里先返回 None，在主流程里处理
    return d_ct

def compute_probabilistic_distances(test_X, model, mapping_matrix):
    """
    计算样本到类别的"概率距离" (解决 One Class -> Many Prototypes 问题)。
    逻辑:
    1. 计算样本到所有 Prototypes 的相似度。
    2. 利用 mapping_matrix (N_proto x N_class) 将相似度聚合到 Class 上。
    3. Distance = 1 - ClassSimilarity
    """
    # 1. 获取 Prototypes 权重
    # 兼容 DataParallel 和不同模型结构
    if hasattr(model, 'module'):
        proto_layer = model.module.prototypes
    else:
        proto_layer = model.prototypes

    if isinstance(proto_layer, torch.nn.ModuleList) or hasattr(proto_layer, 'prototypes0'):
         proto_weight = getattr(proto_layer, 'prototypes0').weight.data.cpu()
    elif isinstance(proto_layer, torch.nn.Linear):
         proto_weight = proto_layer.weight.data.cpu()
    else:
        return None

    # 2. 归一化 (Embedding & Prototypes)
    # SwAV 的核心是在单位球面上计算点积
    test_X_norm = torch.nn.functional.normalize(test_X, p=2, dim=1)
    proto_norm = torch.nn.functional.normalize(proto_weight, p=2, dim=1) # [N_proto, Dim]

    # 3. 计算样本到所有 Prototypes 的相似度 (Proto Scores)
    # [N_samples, N_proto]
    proto_sim = torch.matmul(test_X_norm, proto_norm.t())
    
    # ReLU: 关键步骤！我们只关心正相关的原型，负相关的当做0处理，避免干扰
    proto_sim = torch.nn.functional.relu(proto_sim)

    # 4. 映射到类别 (Class Scores)
    # mapping_matrix: [N_proto, N_class]
    # class_sim: [N_samples, N_class]
    # 这一步实现了加权求和：属于同一个 Class 的所有 Prototypes 的分数会加在一起
    class_sim = torch.matmul(proto_sim, mapping_matrix)

    # 5. 转换为距离
    # 理论上 class_sim 最大可能超过 1 (如果样本同时像多个属于同一类的原型)
    # 但为了兼容 OSR 阈值逻辑，我们限制在 [0, 1] 并转为距离
    class_sim = torch.clamp(class_sim, 0.0, 1.0)
    d_ct = 1.0 - class_sim.numpy()

    return d_ct

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
            embedding = ret[0]
            output = ret[1]
            # Ignore logits/aux_logits for training data feature extraction
            
            # Use embedding (projection head features) instead of output (prototypes/logits)
            feats = embedding.cpu()
            train_features.append(feats)
            train_labels.append(labels)
            
    train_X = torch.cat(train_features, dim=0)
    train_Y = torch.cat(train_labels, dim=0)
    
    num_known = args.num_classes
    
    # 2. Extract features for test data (known + unknown)
    logger.info("Extracting features from test data...")
    test_features = []
    test_embeddings_list = []
    test_labels = []
    test_aux_logits_list = []
    test_intermediate_list = []
    
    # Known test data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            ret = model(inputs)
            
            # Unpack return values based on length
            # Possible returns from WTNet.forward:
            # 3: embedding, proto_out, final_intermediate
            # 4: embedding, proto_out, final_aux_logits, final_intermediate
            # 4: embedding, proto_out, logits, final_intermediate
            # 5: embedding, proto_out, logits, final_aux_logits, final_intermediate
            
            embedding = ret[0]
            output = ret[1] # proto_out
            aux_logits = None
            logits = None
            intermediate = None
            
            if len(ret) == 3:
                intermediate = ret[2]
            elif len(ret) == 4:
                # Check type of 3rd element to distinguish between logits (Tensor) and aux_logits (dict)
                if isinstance(ret[2], dict):
                    aux_logits = ret[2]
                    intermediate = ret[3]
                else:
                    logits = ret[2]
                    intermediate = ret[3]
            elif len(ret) == 5:
                logits = ret[2]
                aux_logits = ret[3]
                intermediate = ret[4]

            if i == 0:
                # logger.info(f"DEBUG: ret length: {len(ret)}")
                pass

            # Use embedding for test_features as well
            test_features.append(embedding.cpu())
            test_embeddings_list.append(embedding.cpu())
            test_labels.append(labels)
            if aux_logits is not None:
                test_aux_logits_list.append({k: v.cpu() for k, v in aux_logits.items()})
            if intermediate is not None:
                test_intermediate_list.append(intermediate.cpu())
            
    # Unknown test data
    if unknown_loader:
        with torch.no_grad():
            for inputs, labels in tqdm(unknown_loader):
                inputs = inputs.to(device)
                ret = model(inputs)
                
                embedding = ret[0]
                output = ret[1]
                aux_logits = None
                logits = None
                intermediate = None
                
                if len(ret) == 3:
                    intermediate = ret[2]
                elif len(ret) == 4:
                    if isinstance(ret[2], dict):
                        aux_logits = ret[2]
                        intermediate = ret[3]
                    else:
                        logits = ret[2]
                        intermediate = ret[3]
                elif len(ret) == 5:
                    logits = ret[2]
                    aux_logits = ret[3]
                    intermediate = ret[4]

                # Use embedding for test_features as well
                test_features.append(embedding.cpu())
                test_embeddings_list.append(embedding.cpu())
                test_labels.append(labels)
                if aux_logits is not None:
                    test_aux_logits_list.append({k: v.cpu() for k, v in aux_logits.items()})
                if intermediate is not None:
                    test_intermediate_list.append(intermediate.cpu())
                
    test_X = torch.cat(test_features, dim=0)
    test_Y = torch.cat(test_labels, dim=0)
    test_E = torch.cat(test_embeddings_list, dim=0)
    
    # Concatenate intermediate features if available
    if test_intermediate_list:
        test_Inter = torch.cat(test_intermediate_list, dim=0)
        # Combine Embedding (Final Layer) with Intermediate Features
        # Normalize both before concatenation to avoid scale dominance
        test_E_norm = torch.nn.functional.normalize(test_E, dim=1)
        test_Inter_norm = torch.nn.functional.normalize(test_Inter, dim=1)
        test_Combined = torch.cat([test_E_norm, test_Inter_norm], dim=1)
        logger.info(f"Using Enhanced Features for Clustering: Dim {test_Combined.shape[1]} (Final {test_E.shape[1]} + Inter {test_Inter.shape[1]})")
    else:
        test_Combined = test_E
        logger.info("Intermediate features not available, using Final Layer only.")

    # 3. Evaluate Euclidean Distance
    logger.info("Evaluating with Euclidean Distance...")
    d_ct_eu, theta_eu = compute_distances(train_X, train_Y, test_X, num_known, metric='euclidean')
    tkr_eu, tur_eu, kp_eu, fkr_eu, mean_acc_eu, label_hat_eu = evaluate_metric(test_Y, d_ct_eu, theta_eu, num_known, "Euclidean Distance")

    # Generate plots
    if hasattr(args, 'dump_path') and args.dump_path:
        try:
            plot_tsne(test_X, test_Y, num_known, args.dump_path)
            plot_distance_histogram(d_ct_eu, test_Y, num_known, "Euclidean Distance", args.dump_path)
        except Exception as e:
            logger.info(f"Failed to generate plots: {e}")



    # --- Stage 2 UP (Clustering-based) for Euclidean ---
    stage2_up_eu_db = None
    stage2_up_eu_sil = None
    stage2_up_eu_kmeans = None
    try:
        logger.info("Computing Stage 2 UP (Euclidean)...")
        # Pass test_Combined instead of test_X for clustering
        res = compute_stage2_up(test_Combined, test_Y, label_hat_eu, theta_eu, num_known)
        stage2_up_eu_db = res['up_db']
        stage2_up_eu_sil = res['up_sil']
        stage2_up_eu_kmeans = res.get('up_kmeans', None)
        logger.info(f"Stage 2 UP (Euclidean) [DB]: {stage2_up_eu_db:.4f}")
        logger.info(f"Stage 2 UP (Euclidean) [Silhouette]: {stage2_up_eu_sil:.4f}")
        if stage2_up_eu_kmeans is not None:
            logger.info(f"Stage 2 UP (Euclidean) [KMeans]: {stage2_up_eu_kmeans:.4f}")
        else:
            logger.info("Stage 2 UP (Euclidean) [KMeans]: None")

        # Unknown per-class classification accuracy (Stage-2)
        try:
            if res.get('unknown_acc_db'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Euclidean) [DB-majority]:")
                for u in sorted(res['unknown_acc_db'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_db'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Euclidean) [DB-majority]: {res.get('unknown_acc_mean_db', 0.0):.4f}")
            if res.get('unknown_acc_sil'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Euclidean) [Sil-majority]:")
                for u in sorted(res['unknown_acc_sil'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_sil'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Euclidean) [Sil-majority]: {res.get('unknown_acc_mean_sil', 0.0):.4f}")
            if res.get('unknown_acc_kmeans'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Euclidean) [KMeans-majority]:")
                for u in sorted(res['unknown_acc_kmeans'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_kmeans'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Euclidean) [KMeans-majority]: {res.get('unknown_acc_mean_kmeans', 0.0):.4f}")
        except Exception:
            pass
    except Exception as e:
        logger.info(f"Stage 2 UP (Euclidean): failed ({e})")

    # 4. Evaluate Mahalanobis Distance
    logger.info("Evaluating with Mahalanobis Distance...")
    d_ct_mahal, theta_mahal = compute_distances(train_X, train_Y, test_X, num_known, metric='mahalanobis')
    tkr_mahal, tur_mahal, kp_mahal, fkr_mahal, mean_acc_mahal, label_hat_mahal = evaluate_metric(test_Y, d_ct_mahal, theta_mahal, num_known, "Mahalanobis Distance")

    # Generate plots
    if hasattr(args, 'dump_path') and args.dump_path:
        try:
            # plot_tsne(test_X, test_Y, num_known, args.dump_path) # Already plotted in Euclidean section
            plot_distance_histogram(d_ct_mahal, test_Y, num_known, "Mahalanobis Distance", args.dump_path)
        except Exception as e:
            logger.info(f"Failed to generate plots: {e}")



    # --- Stage 2 UP (Clustering-based) for Mahalanobis ---
    stage2_up_mahal_db = None
    stage2_up_mahal_sil = None
    stage2_up_mahal_kmeans = None
    try:
        logger.info("Computing Stage 2 UP (Mahalanobis)...")
        # Pass test_Combined instead of test_X for clustering
        res = compute_stage2_up(test_Combined, test_Y, label_hat_mahal, theta_mahal, num_known)
        stage2_up_mahal_db = res['up_db']
        stage2_up_mahal_sil = res['up_sil']
        stage2_up_mahal_kmeans = res.get('up_kmeans', None)
        logger.info(f"Stage 2 UP (Mahalanobis) [DB]: {stage2_up_mahal_db:.4f}")
        logger.info(f"Stage 2 UP (Mahalanobis) [Silhouette]: {stage2_up_mahal_sil:.4f}")
        if stage2_up_mahal_kmeans is not None:
            logger.info(f"Stage 2 UP (Mahalanobis) [KMeans]: {stage2_up_mahal_kmeans:.4f}")
        else:
            logger.info("Stage 2 UP (Mahalanobis) [KMeans]: None")

        # Unknown per-class classification accuracy (Stage-2)
        try:
            if res.get('unknown_acc_db'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Mahalanobis) [DB-majority]:")
                for u in sorted(res['unknown_acc_db'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_db'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Mahalanobis) [DB-majority]: {res.get('unknown_acc_mean_db', 0.0):.4f}")
            if res.get('unknown_acc_sil'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Mahalanobis) [Sil-majority]:")
                for u in sorted(res['unknown_acc_sil'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_sil'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Mahalanobis) [Sil-majority]: {res.get('unknown_acc_mean_sil', 0.0):.4f}")
            if res.get('unknown_acc_kmeans'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Mahalanobis) [KMeans-majority]:")
                for u in sorted(res['unknown_acc_kmeans'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_kmeans'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Mahalanobis) [KMeans-majority]: {res.get('unknown_acc_mean_kmeans', 0.0):.4f}")
        except Exception:
            pass
    except Exception as e:
        logger.info(f"Stage 2 UP (Mahalanobis): failed ({e})")

    # --- Consistency Check Strategy ---
    logger.info("Applying Consistency Check Strategy...")
    # Use Euclidean for known classification, Mahalanobis for unknown rejection
    # If Euclidean says known (label < num_known) AND Mahalanobis says known (dist < theta), accept as known
    # Else reject as unknown
    
    # We need to re-run evaluate_metric with a custom logic or just combine results
    # Here we implement a simple combination:
    # Final Label = Euclidean Label if (Euclidean Label != -1 AND Mahalanobis Label != -1) else -1
    # Wait, the standard consistency check is:
    # If both accept, accept. If one rejects, reject.
    
    # Let's use the labels from previous steps
    # label_hat_eu: -1 if rejected by Euclidean threshold
    # label_hat_mahal: -1 if rejected by Mahalanobis threshold
    
    # But wait, label_hat contains the predicted class if accepted, or -1 if rejected.
    # Consistency: Accept only if both accept AND they agree on the class (optional, but safer)
    # Or just: Accept if both accept.
    
    # Strategy 1: Intersection of Acceptance
    # If label_hat_eu != -1 AND label_hat_mahal != -1:
    #    Final = label_hat_eu (assuming they agree or we trust Euclidean for classification)
    # Else:
    #    Final = -1
    
    label_hat_consistency = label_hat_eu.copy()
    reject_mask = (label_hat_eu == -1) | (label_hat_mahal == -1)
    label_hat_consistency[reject_mask] = -1
    
    # Evaluate Consistency Results
    # We need a dummy distance list for the function API, but it won't be used for thresholding since we already have labels
    # So we can just pass d_ct_eu and a dummy theta, but evaluate_metric re-calculates threshold if we pass distances.
    # We should write a helper that takes labels directly.
    
    # Re-implement metrics calculation for fixed labels
    tkr_c, tur_c, kp_c, fkr_c, mean_acc_c = compute_osr_metrics(test_Y, label_hat_consistency, num_known)
    
    logger.info("--- Consistency Check Results (Euclidean + Aux) ---")
    logger.info(f"TKR: {tkr_c:.4f}, TUR: {tur_c:.4f}, KP: {kp_c:.4f}, FKR: {fkr_c:.4f}")
    logger.info(f"Mean Known Accuracy: {mean_acc_c:.4f}")
    
    # Stage 2 UP for Consistency
    logger.info("Computing Stage 2 UP (Consistency - Euclidean)...")
    try:
        # Pass test_Combined instead of test_X
        res = compute_stage2_up(test_Combined, test_Y, label_hat_consistency, theta_eu, num_known)
        logger.info(f"Stage 2 UP (Consistency - Euclidean) [DB]: {res['up_db']:.4f}")
        logger.info(f"Stage 2 UP (Consistency - Euclidean) [Silhouette]: {res['up_sil']:.4f}")
        if 'up_kmeans' in res:
             logger.info(f"Stage 2 UP (Consistency - Euclidean) [KMeans]: {res['up_kmeans']:.4f}")

        # Unknown per-class classification accuracy (Stage-2)
        try:
            if res.get('unknown_acc_db'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Consistency - Euclidean) [DB-majority]:")
                for u in sorted(res['unknown_acc_db'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_db'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Consistency - Euclidean) [DB-majority]: {res.get('unknown_acc_mean_db', 0.0):.4f}")
            if res.get('unknown_acc_sil'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Consistency - Euclidean) [Sil-majority]:")
                for u in sorted(res['unknown_acc_sil'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_sil'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Consistency - Euclidean) [Sil-majority]: {res.get('unknown_acc_mean_sil', 0.0):.4f}")
            if res.get('unknown_acc_kmeans'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Consistency - Euclidean) [KMeans-majority]:")
                for u in sorted(res['unknown_acc_kmeans'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_kmeans'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Consistency - Euclidean) [KMeans-majority]: {res.get('unknown_acc_mean_kmeans', 0.0):.4f}")
        except Exception:
            pass
    except Exception as e:
        logger.info(f"Stage 2 UP (Consistency): failed ({e})")

    # Strategy 2: Union of Rejection (Same as Intersection of Acceptance)
    # What if we use Mahalanobis for rejection and Euclidean for classification?
    # That is effectively what we did above, but we also required Euclidean to accept.
    # If we trust Mahalanobis for rejection more:
    # Final = label_hat_eu if label_hat_mahal != -1 else -1
    
    label_hat_consistency_2 = label_hat_eu.copy()
    reject_mask_2 = (label_hat_mahal == -1)
    label_hat_consistency_2[reject_mask_2] = -1
    
    tkr_c2, tur_c2, kp_c2, fkr_c2, mean_acc_c2 = compute_osr_metrics(test_Y, label_hat_consistency_2, num_known)
    
    logger.info("--- Consistency Check Results (Mahalanobis + Aux) ---")
    logger.info(f"TKR: {tkr_c2:.4f}, TUR: {tur_c2:.4f}, KP: {kp_c2:.4f}, FKR: {fkr_c2:.4f}")
    logger.info(f"Mean Known Accuracy: {mean_acc_c2:.4f}")
    
    logger.info("Computing Stage 2 UP (Consistency - Mahalanobis)...")
    try:
        # Pass test_Combined instead of test_X
        res = compute_stage2_up(test_Combined, test_Y, label_hat_consistency_2, theta_mahal, num_known)
        logger.info(f"Stage 2 UP (Consistency - Mahalanobis) [DB]: {res['up_db']:.4f}")
        logger.info(f"Stage 2 UP (Consistency - Mahalanobis) [Silhouette]: {res['up_sil']:.4f}")
        if 'up_kmeans' in res:
             logger.info(f"Stage 2 UP (Consistency - Mahalanobis) [KMeans]: {res['up_kmeans']:.4f}")

        # Unknown per-class classification accuracy (Stage-2)
        try:
            if res.get('unknown_acc_db'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Consistency - Mahalanobis) [DB-majority]:")
                for u in sorted(res['unknown_acc_db'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_db'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Consistency - Mahalanobis) [DB-majority]: {res.get('unknown_acc_mean_db', 0.0):.4f}")
            if res.get('unknown_acc_sil'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Consistency - Mahalanobis) [Sil-majority]:")
                for u in sorted(res['unknown_acc_sil'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_sil'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Consistency - Mahalanobis) [Sil-majority]: {res.get('unknown_acc_mean_sil', 0.0):.4f}")
            if res.get('unknown_acc_kmeans'):
                logger.info("Stage 2 Unknown Per-class Accuracy (Consistency - Mahalanobis) [KMeans-majority]:")
                for u in sorted(res['unknown_acc_kmeans'].keys()):
                    logger.info(f"  Unknown Class {u}: {res['unknown_acc_kmeans'][u]:.4f}")
                logger.info(f"Stage 2 Mean Unknown Acc (Consistency - Mahalanobis) [KMeans-majority]: {res.get('unknown_acc_mean_kmeans', 0.0):.4f}")
        except Exception:
            pass
    except Exception as e:
        logger.info(f"Stage 2 UP (Consistency 2): failed ({e})")

    # 5. Return combined results
    results = {
        'tkr_eu': tkr_eu, 'tur_eu': tur_eu, 'kp_eu': kp_eu, 'fkr_eu': fkr_eu, 'mean_acc_eu': mean_acc_eu,
        'tkr_mahal': tkr_mahal, 'tur_mahal': tur_mahal, 'kp_mahal': kp_mahal, 'fkr_mahal': fkr_mahal, 'mean_acc_mahal': mean_acc_mahal,
        'label_hat_eu': label_hat_eu, 'label_hat_mahal': label_hat_mahal,
    'stage2_up_eu_db': stage2_up_eu_db if 'stage2_up_eu_db' in locals() else None,
    'stage2_up_eu_sil': stage2_up_eu_sil if 'stage2_up_eu_sil' in locals() else None,
    'stage2_up_eu_kmeans': stage2_up_eu_kmeans if 'stage2_up_eu_kmeans' in locals() else None,
    'stage2_up_mahal_db': stage2_up_mahal_db if 'stage2_up_mahal_db' in locals() else None,
    'stage2_up_mahal_sil': stage2_up_mahal_sil if 'stage2_up_mahal_sil' in locals() else None,
    'stage2_up_mahal_kmeans': stage2_up_mahal_kmeans if 'stage2_up_mahal_kmeans' in locals() else None,
        # NOTE: consistency-check UP values are logged above but not necessarily
        # stored as separate variables. Keep these keys for backward-compat,
        # but default to None to avoid NameError/static analysis issues.
        'stage2_up_cc_db': locals().get('stage2_up_cc_db', None),
        'stage2_up_cc_sil': locals().get('stage2_up_cc_sil', None),
        'stage2_up_cc_kmeans': locals().get('stage2_up_cc_kmeans', None),
        'stage2_up_cc_m_db': locals().get('stage2_up_cc_m_db', None),
        'stage2_up_cc_m_sil': locals().get('stage2_up_cc_m_sil', None),
        'stage2_up_cc_m_kmeans': locals().get('stage2_up_cc_m_kmeans', None)
    }
    
    # === Evaluate with Probabilistic Prototype Mapping (解决撞车问题的终极方案) ===
    logger.info("Evaluating with Probabilistic Prototype Mapping...")

    # --- A. 自动构建映射矩阵 W [N_proto, N_class] ---
    # 1. 获取原型权重
    if hasattr(model, 'module'):
        proto_layer_ref = model.module.prototypes
    else:
        proto_layer_ref = model.prototypes
        
    if isinstance(proto_layer_ref, torch.nn.ModuleList) or hasattr(proto_layer_ref, 'prototypes0'):
         proto_weight = getattr(proto_layer_ref, 'prototypes0').weight.data.cpu()
    else:
         proto_weight = proto_layer_ref.weight.data.cpu()
    
    n_protos = proto_weight.shape[0]
    # mapping_matrix[p, c] 表示 Prototype p 属于 Class c 的概率/权重
    mapping_matrix = torch.zeros(n_protos, num_known)
    
    # 2. 跑一遍训练集，统计每个样本被分配给了哪个 Prototype
    # 注意：这里我们使用 Projection Head 的特征 (Embedding)
    train_X_norm = torch.nn.functional.normalize(train_X, p=2, dim=1)
    proto_norm = torch.nn.functional.normalize(proto_weight, p=2, dim=1)
    
    # 计算每个训练样本最近的原型
    # [N_train, N_proto]
    sim = torch.matmul(train_X_norm, proto_norm.t())
    assigned_protos = torch.argmax(sim, dim=1) # [N_train]
    
    # 3. 填充计数矩阵
    for i in range(train_Y.shape[0]):
        pid = assigned_protos[i].item()
        cid = train_Y[i].item()
        # 确保只处理已知类 (train_Y 理论上都是已知类，但为了保险)
        if cid < num_known: 
            mapping_matrix[pid, cid] += 1.0
            
    # 4. 归一化矩阵 (Row Normalization: P(Class | Proto))
    # 每一行代表一个 Prototype，它的能量应该分配给哪些 Class
    row_sums = mapping_matrix.sum(dim=1, keepdim=True)
    # 避免除以 0 (如果有些死掉的原型没分到任何样本，就保持 0)
    row_sums[row_sums == 0] = 1.0 
    mapping_matrix = mapping_matrix / row_sums
    
    # (调试信息) 打印一下矩阵信息，确认是否解决了您遇到的 Class 0/18 撞车问题
    # 如果 P39 同时服务于 C18 和 C0，这里会显示出来
    if n_protos > 39 and num_known > 18:
        if mapping_matrix[39, 18] > 0 and mapping_matrix[39, 0] > 0:
            logger.info(f"✔ Collision Auto-Resolved: Proto 39 -> Class 18 ({mapping_matrix[39,18]:.2f}), Class 0 ({mapping_matrix[39,0]:.2f})")
        else:
            logger.info("Info: Proto 39 mapping check - No mixed assignment found or indices differ.")

    # --- B. 计算距离并评估 ---
    
    # 1. 计算训练集距离 (用于定阈值 Theta)
    # 注意：现在使用的是 mapping_matrix 计算出的加权距离
    d_train = compute_probabilistic_distances(train_X, model, mapping_matrix)
    
    if d_train is not None:
        theta_prob = np.zeros(num_known)
        for clas in range(num_known):
            mask = (train_Y == clas).numpy()
            if mask.sum() > 0:
                # 取属于该类的样本，在该类上的距离
                own_class_dist = d_train[mask, clas]
                theta_prob[clas] = outlier_check(own_class_dist)
                
        # 2. 计算测试集距离
        d_test = compute_probabilistic_distances(test_X, model, mapping_matrix)
        
        # 3. 评估指标
        # 这里的 evaluate_metric 会自动处理 Open Set 拒识逻辑
        tkr_p, tur_p, kp_p, fkr_p, mean_acc_p, label_hat_p = evaluate_metric(
            test_Y, d_test, torch.tensor(theta_prob), num_known, "SwAV Probabilistic"
        )
        
        # 4. (可选) 绘制直方图
        if hasattr(args, 'dump_path') and args.dump_path:
             plot_distance_histogram(d_test, test_Y, num_known, "Probabilistic Cosine", args.dump_path)
             
        # 5. (可选) 计算 Stage 2 UP 指标
        logger.info("Computing Stage 2 UP (Probabilistic)...")
        try:
            # 依然使用 Enhanced Features (Combined) 进行聚类，但使用新的 label_hat 进行筛选
            res = compute_stage2_up(test_Combined, test_Y, label_hat_p, theta_prob, num_known)
            logger.info(f"Stage 2 UP (Probabilistic) [DB]: {res['up_db']:.4f}")
            logger.info(f"Stage 2 UP (Probabilistic) [Silhouette]: {res['up_sil']:.4f}")
            if 'up_kmeans' in res:
                logger.info(f"Stage 2 UP (Probabilistic) [KMeans]: {res['up_kmeans']:.4f}")
        except Exception as e:
            logger.info(f"Stage 2 UP (Probabilistic): failed ({e})")
    
    return results
