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
        return {'up_db': 0.0, 'up_sil': 0.0, 'up_kmeans': 0.0}

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
            logger.info("Stage 2 UP: u=1 detected (Compact). Proceeding to clustering to find substructure.")
            # Calculate UP (Precision of Unknowns)
            # a: number of true unknowns in predicted unknowns
            # a = np.sum(test_Y_normalized[unknown_mask] == -1)
            # c = predict_unknown_X.shape[0]
            # up = a / c if c > 0 else 0.0
            # return {'up_db': up, 'up_sil': up, 'up_kmeans': up}
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
            return {'up_db': 0.0, 'up_sil': 0.0, 'up_kmeans': 0.0}

        # Select k
        u_db = candidates[np.argmin(db_scores)]
        u_sil = candidates[np.argmax(sil_scores)]
        
        logger.info(f"Stage 2 UP: Selected u_db={u_db} (DB Score), u_sil={u_sil} (Silhouette Score)")
        
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
                                
            return np.sum(dominate_sample) / predict_unknown_X.shape[0]

        up_db = calculate_up_for_k(u_db)
        up_sil = calculate_up_for_k(u_sil)

        # KMeans method: use k = num_unknown (fallback to 2), capped by sample count
        n_samples = predict_unknown_X_scaled.shape[0]
        k_kmeans = num_unknown if num_unknown >= 2 else 2
        k_kmeans = min(k_kmeans, max(2, n_samples))
        if k_kmeans < 2 or n_samples < 2:
            up_kmeans = 0.0
        else:
            up_kmeans = calculate_up_for_k(int(k_kmeans))

        return {'up_db': up_db, 'up_sil': up_sil, 'up_kmeans': up_kmeans}

    except Exception as e:
        logger.info(f"Stage 2 UP: Clustering logic failed ({e})")
        return {'up_db': 0.0, 'up_sil': 0.0, 'up_kmeans': 0.0}

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
    test_embeddings_list = []
    test_aux_logits_list = []
    
    # Known test data
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            ret = model(inputs)
            
            embedding = ret[0]
            output = ret[1]
            aux_logits = None
            if len(ret) == 3:
                if isinstance(ret[2], dict):
                    aux_logits = ret[2]
            elif len(ret) == 4:
                aux_logits = ret[3]

            if i == 0:
                # logger.info(f"DEBUG: ret length: {len(ret)}")
                pass

            test_features.append(output.cpu())
            test_embeddings_list.append(embedding.cpu())
            test_labels.append(labels)
            if aux_logits is not None:
                test_aux_logits_list.append({k: v.cpu() for k, v in aux_logits.items()})
            
    # Unknown test data
    if unknown_loader:
        with torch.no_grad():
            for inputs, labels in tqdm(unknown_loader):
                inputs = inputs.to(device)
                ret = model(inputs)
                
                embedding = ret[0]
                output = ret[1]
                aux_logits = None
                if len(ret) == 3:
                    if isinstance(ret[2], dict):
                        aux_logits = ret[2]
                elif len(ret) == 4:
                    aux_logits = ret[3]

                test_features.append(output.cpu())
                test_embeddings_list.append(embedding.cpu())
                test_labels.append(labels)
                if aux_logits is not None:
                    test_aux_logits_list.append({k: v.cpu() for k, v in aux_logits.items()})
                
    test_X = torch.cat(test_features, dim=0)
    test_Y = torch.cat(test_labels, dim=0)
    test_E = torch.cat(test_embeddings_list, dim=0)
    
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
        res = compute_stage2_up(test_X, test_Y, label_hat_eu, theta_eu, num_known)
        stage2_up_eu_db = res['up_db']
        stage2_up_eu_sil = res['up_sil']
        stage2_up_eu_kmeans = res.get('up_kmeans', None)
        logger.info(f"Stage 2 UP (Euclidean) [DB]: {stage2_up_eu_db:.4f}")
        logger.info(f"Stage 2 UP (Euclidean) [Silhouette]: {stage2_up_eu_sil:.4f}")
        if stage2_up_eu_kmeans is not None:
            logger.info(f"Stage 2 UP (Euclidean) [KMeans]: {stage2_up_eu_kmeans:.4f}")
        else:
            logger.info("Stage 2 UP (Euclidean) [KMeans]: None")
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
        res = compute_stage2_up(test_X, test_Y, label_hat_mahal, theta_mahal, num_known)
        stage2_up_mahal_db = res['up_db']
        stage2_up_mahal_sil = res['up_sil']
        stage2_up_mahal_kmeans = res.get('up_kmeans', None)
        logger.info(f"Stage 2 UP (Mahalanobis) [DB]: {stage2_up_mahal_db:.4f}")
        logger.info(f"Stage 2 UP (Mahalanobis) [Silhouette]: {stage2_up_mahal_sil:.4f}")
        if stage2_up_mahal_kmeans is not None:
            logger.info(f"Stage 2 UP (Mahalanobis) [KMeans]: {stage2_up_mahal_kmeans:.4f}")
        else:
            logger.info("Stage 2 UP (Mahalanobis) [KMeans]: None")
    except Exception as e:
        logger.info(f"Stage 2 UP (Mahalanobis): failed ({e})")

    # --- Consistency Check Strategy ---
    if test_aux_logits_list:
        logger.info("Applying Consistency Check Strategy...")
        
        # Concatenate aux logits
        keys = test_aux_logits_list[0].keys()
        aux_preds = {}
        for k in keys:
            logits_k = torch.cat([d[k] for d in test_aux_logits_list], dim=0)
            aux_preds[k] = torch.argmax(logits_k, dim=1).numpy()
            
        # Apply consistency
        refined_label_hat = label_hat_eu.copy()
        
        for k in keys:
            # If main prediction is known (>=0), it must match branch prediction
            # If main prediction is unknown (-1), it remains unknown
            disagreement = (refined_label_hat != -1) & (refined_label_hat != aux_preds[k])
            refined_label_hat[disagreement] = -1
            
        # Re-evaluate metrics
        test_Y_normalized = test_Y.numpy().copy()
        test_Y_normalized[test_Y_normalized >= num_known] = -1
        
        tkr_cc, tur_cc, kp_cc, fkr_cc, accuracy_cc = metrics_stage_1(test_Y_normalized, refined_label_hat, num_known)
        
        logger.info(f"--- Consistency Check Results (Euclidean + Aux) ---")
        logger.info(f"TKR: {tkr_cc:.4f}, TUR: {tur_cc:.4f}, KP: {kp_cc:.4f}, FKR: {fkr_cc:.4f}")
        logger.info(f"Mean Known Accuracy: {np.mean(accuracy_cc):.4f}")
        
        # Stage 2 UP for Consistency Check (Euclidean)
        stage2_up_cc_db = None
        stage2_up_cc_sil = None
        stage2_up_cc_kmeans = None
        try:
            logger.info("Computing Stage 2 UP (Consistency - Euclidean)...")
            res = compute_stage2_up(test_X, test_Y, refined_label_hat, theta_eu, num_known)
            stage2_up_cc_db = res['up_db']
            stage2_up_cc_sil = res['up_sil']
            stage2_up_cc_kmeans = res.get('up_kmeans', None)
            logger.info(f"Stage 2 UP (Consistency - Euclidean) [DB]: {stage2_up_cc_db:.4f}")
            logger.info(f"Stage 2 UP (Consistency - Euclidean) [Silhouette]: {stage2_up_cc_sil:.4f}")
            if stage2_up_cc_kmeans is not None:
                logger.info(f"Stage 2 UP (Consistency - Euclidean) [KMeans]: {stage2_up_cc_kmeans:.4f}")
            else:
                logger.info("Stage 2 UP (Consistency - Euclidean) [KMeans]: None")
        except Exception as e:
            logger.info(f"Stage 2 UP (Consistency - Euclidean): failed ({e})")

        # Apply consistency for Mahalanobis
        refined_label_hat_mahal = label_hat_mahal.copy()
        
        for k in keys:
            disagreement = (refined_label_hat_mahal != -1) & (refined_label_hat_mahal != aux_preds[k])
            refined_label_hat_mahal[disagreement] = -1
            
        tkr_cc_m, tur_cc_m, kp_cc_m, fkr_cc_m, accuracy_cc_m = metrics_stage_1(test_Y_normalized, refined_label_hat_mahal, num_known)
        
        logger.info(f"--- Consistency Check Results (Mahalanobis + Aux) ---")
        logger.info(f"TKR: {tkr_cc_m:.4f}, TUR: {tur_cc_m:.4f}, KP: {kp_cc_m:.4f}, FKR: {fkr_cc_m:.4f}")
        logger.info(f"Mean Known Accuracy: {np.mean(accuracy_cc_m):.4f}")
        
        # Stage 2 UP for Consistency Check (Mahalanobis)
        stage2_up_cc_m_db = None
        stage2_up_cc_m_sil = None
        stage2_up_cc_m_kmeans = None
        try:
            logger.info("Computing Stage 2 UP (Consistency - Mahalanobis)...")
            res = compute_stage2_up(test_X, test_Y, refined_label_hat_mahal, theta_mahal, num_known)
            stage2_up_cc_m_db = res['up_db']
            stage2_up_cc_m_sil = res['up_sil']
            stage2_up_cc_m_kmeans = res.get('up_kmeans', None)
            logger.info(f"Stage 2 UP (Consistency - Mahalanobis) [DB]: {stage2_up_cc_m_db:.4f}")
            logger.info(f"Stage 2 UP (Consistency - Mahalanobis) [Silhouette]: {stage2_up_cc_m_sil:.4f}")
            if stage2_up_cc_m_kmeans is not None:
                logger.info(f"Stage 2 UP (Consistency - Mahalanobis) [KMeans]: {stage2_up_cc_m_kmeans:.4f}")
            else:
                logger.info("Stage 2 UP (Consistency - Mahalanobis) [KMeans]: None")
        except Exception as e:
            logger.info(f"Stage 2 UP (Consistency - Mahalanobis): failed ({e})")

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
        'stage2_up_cc_db': stage2_up_cc_db if 'stage2_up_cc_db' in locals() else None,
        'stage2_up_cc_sil': stage2_up_cc_sil if 'stage2_up_cc_sil' in locals() else None,
        'stage2_up_cc_kmeans': stage2_up_cc_kmeans if 'stage2_up_cc_kmeans' in locals() else None,
        'stage2_up_cc_m_db': stage2_up_cc_m_db if 'stage2_up_cc_m_db' in locals() else None,
        'stage2_up_cc_m_sil': stage2_up_cc_m_sil if 'stage2_up_cc_m_sil' in locals() else None,
        'stage2_up_cc_m_kmeans': stage2_up_cc_m_kmeans if 'stage2_up_cc_m_kmeans' in locals() else None
    }
    
    return results
