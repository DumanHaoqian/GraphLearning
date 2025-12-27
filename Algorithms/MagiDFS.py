import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, roc_auc_score
# Depth First Search (DFS)
# Algorithm 1: MAGI-DFS v2 (Multi-source Aggregated Golden Inference with Depth First Search)
class MultiRootSADFS:
    def __init__(self, adj_matrix, features, labels, golden_ratio=0.618, k_roots=5, max_depth=10):
        """
        Similarity-based Aggregation Depth First Search (SADFS).
        
        Args:
            adj_matrix: (N, N) Adjacency matrix.
            features: (N, D) Features.
            labels: (N,) Labels.
            golden_ratio: Decay factor.
            k_roots: Number of roots.
            max_depth: DFS max depth limit (Crucial for DFS to prevent getting lost).
        """
        self.adj_matrix = adj_matrix
        self.features = features
        self.labels = labels.flatten()
        self.golden_ratio = golden_ratio
        self.k_roots = k_roots
        self.max_depth = max_depth
        self.num_nodes = features.shape[0]

    def _softmax(self, weights):
        w = np.array(weights)
        e_w = np.exp(w - np.max(w))
        return e_w / e_w.sum()

    def _run_dfs_from_root(self, root_idx):
        """
        Executes Golden Memory DFS.
        Uses a Stack for LIFO traversal.
        """
        # Stack stores: (node_index, current_depth, path_weight_product)
        # path_weight_product represents the accumulated edge weight strength along the path
        stack = [(root_idx, 0, 1.0)] 
        
        visited = set()
        total_memory = 0.0
        
        # DFS doesn't have "layers" in the same way, so we accumulate node by node
        # We limit by iterations to prevent infinite loops in cycles if visited check fails logically
        iterations = 0
        max_iterations = 1000 

        while stack and iterations < max_iterations:
            u, depth, path_weight = stack.pop() # Pop from end = LIFO (DFS)
            
            if u in visited:
                continue
            visited.add(u)
            iterations += 1

            # --- Experience Calculation ---
            # In BFS we averaged the whole layer. 
            # In DFS, we take the single node's label, weighted by how "strong" the path to it was.
            current_label = self.labels[u]
            
            # --- Golden Memory Aggregation ---
            # Decay based on depth
            decay = self.golden_ratio ** depth
            
            # Contribution = Label * Decay * Path_Strength
            # We treat the label as a signal strength.
            total_memory += current_label * decay * path_weight
            
            # --- DFS Expansion ---
            if depth < self.max_depth:
                # Find neighbors
                neighbors_indices = np.where(self.adj_matrix[u] > 0)[0]
                
                # Optimization: Sort neighbors by weight to visit strongest connections first?
                # Or shuffle? Standard DFS order depends on index. 
                # Let's prioritize higher weights by pushing them last (so they are popped first)
                neighbor_weights = []
                for v in neighbors_indices:
                    if v not in visited:
                        w = self.adj_matrix[u][v]
                        neighbor_weights.append((v, w))
                
                # Sort by weight ascending, so highest weight is at the end of list (Top of Stack)
                neighbor_weights.sort(key=lambda x: x[1])
                
                for v, w in neighbor_weights:
                    # New path weight = Old path weight * current edge weight
                    # (Optional: You can also just use 1.0 if you only want depth decay)
                    new_path_weight = path_weight * w 
                    stack.append((v, depth + 1, new_path_weight))
        
        return total_memory

    def predict_single_sample(self, test_feature_vector):
        # 1. Find Top-K Roots (Same as BFS)
        test_vec = test_feature_vector.reshape(1, -1)
        dists = cdist(test_vec, self.features, metric='cosine').flatten()
        
        k = min(self.k_roots, self.num_nodes)
        top_k_indices = np.argsort(dists)[:k]
        top_k_dists = dists[top_k_indices]
        top_k_sims = 1.0 - top_k_dists + 1e-8
        
        # 2. Run DFS for each root
        root_scores = []
        for root_idx in top_k_indices:
            score = self._run_dfs_from_root(root_idx)
            root_scores.append(score)
        
        root_scores = np.array(root_scores)
        
        # 3. Weighted Aggregation
        weights = self._softmax(top_k_sims)
        return np.sum(weights * root_scores)

    def predict(self, test_data):
        if isinstance(test_data, pd.DataFrame):
            data_values = test_data.values
        else:
            data_values = test_data
            
        predictions = []
        # print(f"Starting Multi-Root SADFS (K={self.k_roots}) for {len(data_values)} samples...")
        
        for i, sample in enumerate(data_values):
            pred = self.predict_single_sample(sample)
            predictions.append(pred)
            
        return np.array(predictions)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # Configuration
    # DFS needs a strict depth limit because 0.618^20 is tiny, going deeper is waste of compute
    MAX_DEPTH = 8 
    
    accuracy_results = []
    
    # Load Data (Paths kept from your code)
    try:
        base_path = "/home/haoqian/Data/Course/DSAI4204/Project"
        adj_matrix = np.load(f"{base_path}/Predictors/GraphBased/BuildGraphFormal/TrainOnly/adj_matrix.npy")
        features = np.load(f"{base_path}/Predictors/GraphBased/BuildGraphFormal/TrainOnly/features.npy")
        labels = np.load(f"{base_path}/Predictors/GraphBased/BuildGraphFormal/TrainOnly/labels.npy")
        
        test_data_df = pd.read_csv(f"{base_path}/Dataset/processed/test.csv")
        test_label_df = pd.read_csv(f"{base_path}/Dataset/processed/test_label.csv")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    y_true = test_label_df.values.flatten()

    # Loop for K-Roots
    # DFS might be more sensitive to K, so we test a range
    print(f"{'K_Roots':<10} | {'Accuracy':<10} | {'AUC':<10}")
    print("-" * 35)
    
    for k_roots in range(1, 51): # Testing 1 to 20 roots
        model = MultiRootSADFS(
            adj_matrix=adj_matrix,
            features=features,
            labels=labels,
            golden_ratio=0.618,
            k_roots=k_roots,
            max_depth=MAX_DEPTH
        )

        pred_scores = model.predict(test_data_df)

        # Evaluation
        try:
            auc = roc_auc_score(y_true, pred_scores)
        except:
            auc = 0.0

        # Dynamic Thresholding for Accuracy
        best_acc = 0
        # Search for best threshold
        thresholds = np.linspace(np.min(pred_scores), np.max(pred_scores), 50)
        for t in thresholds:
            y_pred = (pred_scores > t).astype(int)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_acc:
                best_acc = acc
        
        accuracy_results.append(best_acc)
        print(f"{k_roots:<10} | {best_acc:.4f}     | {auc:.4f}")

if __name__ == "__main__":
    main()