import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from logging import getLogger

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
            label_hat[xi] = -1 # Unknown
        else:
            label_hat[xi] = np.argmin(x_ct[xi])
            
    test_Y_normalized = test_Y.numpy().copy()
    test_Y_normalized[test_Y_normalized >= num_known] = -1
    
    tkr, tur, kp, fkr, accuracy = metrics_stage_1(test_Y_normalized, label_hat, num_known)
    
    logger.info(f"--- {metric_name} Results ---")
    logger.info(f"TKR: {tkr:.4f}, TUR: {tur:.4f}, KP: {kp:.4f}, FKR: {fkr:.4f}")
    logger.info(f"Mean Known Accuracy: {np.mean(accuracy):.4f}")
    return tkr, tur, kp, fkr, np.mean(accuracy)

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
    # Known classes use a discrete colormap (tab20). Unknown classes are plotted
    # like known classes (filled circles) but use a separate colormap and a black
    # edge so they remain distinguishable when overlapping with known points.
    plt.figure(figsize=(12, 10))

    # Plot knowns
    if known_mask.sum() > 0:
        scatter_known = plt.scatter(
            X_embedded[known_mask, 0],
            X_embedded[known_mask, 1],
            c=test_Y_np[known_mask],
            cmap='tab20',
            s=20,
            alpha=0.8,
            label='Known',
            linewidths=0
        )
        # Add a colorbar for known classes
        plt.colorbar(scatter_known, label='Known Class ID')

    # Plot unknowns: map unknown class ids to a dense 0..K-1 range so we can use
    # a separate colormap. We use a distinct colormap (try 'tab20b' then fallback)
    # and draw a thin black edge around markers to help visual separation on overlap.
    if unknown_mask.sum() > 0:
        unknown_labels = test_Y_np[unknown_mask]
        unique_unknowns, unknown_inverse = np.unique(unknown_labels, return_inverse=True)
        n_unknown = len(unique_unknowns)

        # Choose a distinct colormap for unknowns; fall back to Dark2 if tab20b not available
        try:
            unknown_cmap = plt.get_cmap('tab20b')
        except Exception:
            unknown_cmap = plt.get_cmap('Dark2')

        scatter_unknown = plt.scatter(
            X_embedded[unknown_mask, 0],
            X_embedded[unknown_mask, 1],
            c=unknown_inverse,
            cmap=unknown_cmap,
            s=36,
            alpha=0.95,
            edgecolors='k',
            linewidths=0.25,
            label='Unknown'
        )

        # Create a colorbar for unknowns and label ticks with the original class ids
        cbar_unk = plt.colorbar(scatter_unknown, label='Unknown Class ID (original)')
        if n_unknown <= 30:
            ticks = np.arange(n_unknown)
            cbar_unk.set_ticks(ticks)
            cbar_unk.set_ticklabels([str(int(x)) for x in unique_unknowns])
        else:
            # Too many unknowns: show only endpoints
            cbar_unk.set_ticks([0, n_unknown - 1])
            cbar_unk.set_ticklabels([str(int(unique_unknowns[0])), str(int(unique_unknowns[-1]))])

    plt.title("t-SNE: Known vs Unknown", fontsize=16)
    plt.legend(fontsize=12)
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
    evaluate_metric(test_Y, d_ct_eu, theta_eu, num_known, "Euclidean Distance")
    
    # 4. Evaluate Mahalanobis Distance
    logger.info("Evaluating with Mahalanobis Distance...")
    d_ct_ma, theta_ma = compute_distances(train_X, train_Y, test_X, num_known, metric='mahalanobis')
    evaluate_metric(test_Y, d_ct_ma, theta_ma, num_known, "Mahalanobis Distance")
    
    # 5. t-SNE Plotting
    plot_tsne(test_X, test_Y, num_known, args.dump_path)
    
    # 6. Distance Histogram
    plot_distance_histogram(d_ct_eu, test_Y, num_known, "Euclidean Distance", args.dump_path)
    plot_distance_histogram(d_ct_ma, test_Y, num_known, "Mahalanobis Distance", args.dump_path)
    
    logger.info("Evaluation complete. Plots saved to dump path.")
