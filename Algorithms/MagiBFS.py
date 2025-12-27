# Algorithm 1: MAGI (Multi-source Aggregated Golden Inference)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, roc_auc_score
# Breadth First Search (BFS)
# Algorithm 1: MAGI-BFS v1 (Multi-source Aggregated Golden Inference with Breadth First Search)
class MultiRootSABFS:
    def __init__(self, adj_matrix, features, labels, golden_ratio=0.618, k_roots=5, convergence_threshold=1e-5):
        """
        Similarity-based Aggregation Beard First Search (SABFS) with Multi-Root Strategy.
        
        Args:
            adj_matrix: (N, N) Adjacency matrix of the training graph.
            features: (N, D) Features of training nodes.
            labels: (N,) Labels of training nodes.
            golden_ratio: Decay factor (phi), default 0.618.
            k_roots: Number of top similar nodes to use as roots (K-NN).
            convergence_threshold: Threshold to stop BFS if memory update is negligible.
        """
        self.adj_matrix = adj_matrix
        self.features = features
        self.labels = labels.flatten()
        self.golden_ratio = golden_ratio
        self.k_roots = k_roots
        self.threshold = convergence_threshold
        self.num_nodes = features.shape[0]

    def _softmax(self, weights):
        """Compute numerically stable softmax."""
        w = np.array(weights)
        e_w = np.exp(w - np.max(w)) # Subtract max for stability
        return e_w / e_w.sum()

    def _run_bfs_from_root(self, root_idx):
        """
        Executes the Golden Memory BFS starting from a specific root node.
        Returns the accumulated Memory score for this specific tree.
        """
        # Initialization
        visited = set()
        visited.add(root_idx)
        
        # Current layer stores tuples: (node_index, weight_from_parent)
        # For the root (Layer 0), there is no parent weight, so we use None or 0.
        current_layer_nodes = [(root_idx, 0.0)] 
        
        total_memory = 0.0
        prev_memory = -1.0 # To check convergence
        layer_depth = 0
        
        while current_layer_nodes:
            # --- Step 3: Experience Calculation ---
            
            if layer_depth == 0:
                # Layer 0: The root itself. Experience is just its label.
                layer_experience = self.labels[root_idx]
            else:
                # Layer k > 0: Calculate weighted average based on incoming edge weights
                layer_weights = [item[1] for item in current_layer_nodes]
                layer_node_indices = [item[0] for item in current_layer_nodes]
                
                # Softmax Normalization
                alpha = self._softmax(layer_weights)
                
                # Weighted Aggregation of Labels
                current_labels = self.labels[layer_node_indices]
                layer_experience = np.sum(alpha * current_labels)
            
            # --- Step 4: Golden Memory Aggregation ---
            # Memory += Experience * (phi ^ k)
            decay = self.golden_ratio ** layer_depth
            total_memory += layer_experience * decay
            
            # --- Step 5: Convergence Check ---
            if layer_depth > 0 and abs(total_memory - prev_memory) < self.threshold:
                break
            
            prev_memory = total_memory
            
            # --- Prepare Next Layer (BFS Expansion) ---
            next_layer_nodes = []
            
            # Expand from all nodes in current layer
            for u, _ in current_layer_nodes:
                # Find neighbors in Adjacency Matrix
                # adj_matrix[u] gives the row of weights
                neighbors_indices = np.where(self.adj_matrix[u] > 0)[0]
                
                for v in neighbors_indices:
                    if v not in visited:
                        visited.add(v)
                        weight = self.adj_matrix[u][v]
                        next_layer_nodes.append((v, weight))
            
            # Update pointers for next iteration
            current_layer_nodes = next_layer_nodes
            layer_depth += 1
            
            # Safety break for very deep graphs (optional)
            if layer_depth > 20:
                break
                
        return total_memory

    def predict_single_sample(self, test_feature_vector):
        """
        Predicts score for a single test sample using Multi-Root strategy.
        1. Find Top-K most similar training nodes.
        2. Run BFS from each.
        3. Weighted average by similarity (closer roots contribute more).
        """
        # Reshape for cdist (1, D)
        test_vec = test_feature_vector.reshape(1, -1)
        
        # Calculate Cosine Distance (dist = 1 - similarity)
        # We want the smallest distance
        dists = cdist(test_vec, self.features, metric='cosine').flatten()
        
        # Get indices of the Top-K smallest distances (Most similar)
        # If K > N, take all N
        k = min(self.k_roots, self.num_nodes)
        top_k_indices = np.argsort(dists)[:k]
        top_k_dists = dists[top_k_indices]
        
        # Convert distances to similarities (similarity = 1 - distance)
        # Add small epsilon to avoid division issues
        top_k_sims = 1.0 - top_k_dists + 1e-8
        
        # Run BFS for each root and collect scores
        root_scores = []
        for root_idx in top_k_indices:
            score = self._run_bfs_from_root(root_idx)
            root_scores.append(score)
        
        root_scores = np.array(root_scores)
        
        # Similarity-weighted aggregation (closer roots have more influence)
        weights = self._softmax(top_k_sims)
        return np.sum(weights * root_scores)

    def predict(self, test_data):
        """
        Batch prediction for test dataset.
        """
        if isinstance(test_data, pd.DataFrame):
            data_values = test_data.values
        else:
            data_values = test_data
            
        predictions = []
        print(f"Starting Multi-Root SABFS (K={self.k_roots}) for {len(data_values)} samples...")
        
        for i, sample in enumerate(data_values):
            pred = self.predict_single_sample(sample)
            predictions.append(pred)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(data_values)}...")
                
        return np.array(predictions)

# ==============================================================================
# Main Execution Pipeline
# ==============================================================================

def main():
    accuracy = []
    top = 100
    for k_roots in range(1,top):
        print(f"Running with k_roots={k_roots}...")
        # 1. Load Data
        print("Loading datasets...")
        try:
            adj_matrix = np.load("/home/haoqian/Data/Course/DSAI4204/Project/Predictors/GraphBased/BuildGraphFormal/TrainOnly/adj_matrix.npy")
            features = np.load("/home/haoqian/Data/Course/DSAI4204/Project/Predictors/GraphBased/BuildGraphFormal/TrainOnly/features.npy")
            labels = np.load("/home/haoqian/Data/Course/DSAI4204/Project/Predictors/GraphBased/BuildGraphFormal/TrainOnly/labels.npy")
            
            test_data_df = pd.read_csv("/home/haoqian/Data/Course/DSAI4204/Project/Dataset/processed/test.csv")
            test_label_df = pd.read_csv("/home/haoqian/Data/Course/DSAI4204/Project/Dataset/processed/test_label.csv")
            
            print(f"Training Graph Loaded: {adj_matrix.shape} Nodes")
            print(f"Test Data Loaded: {test_data_df.shape}")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        # 2. Initialize Model with Multi-Root Strategy
        # We set k_roots=5 as suggested to improve stability against noise
        sabfs_model = MultiRootSABFS(
            adj_matrix=adj_matrix,
            features=features,
            labels=labels,
            golden_ratio=0.618,
            k_roots=k_roots,  # Top-k Closest Nodes
            convergence_threshold=1e-3
        )

        # 3. Predict
        # Note: Ensure test_data_df only contains features (no ID columns)
        # If there are ID columns, drop them here: e.g., test_data_df.drop(columns=['id'], inplace=True)
        pred_scores = sabfs_model.predict(test_data_df)

        # 4. Evaluation
        y_true = test_label_df.values.flatten()
        
        # Since the output is a continuous accumulated "Memory" score, 
        # we can evaluate it using AUC directly.
        try:
            auc = roc_auc_score(y_true, pred_scores)
            print(f"\nROC AUC Score: {auc:.4f}")
        except ValueError:
            print("\nCould not calculate AUC (possibly only one class in test data).")

        # For Accuracy, we need a threshold. 
        # Since labels are 0/1, and Memory is a weighted sum of labels,
        # the score roughly represents the probability/confidence of being class 1.
        # However, due to the accumulation sum, the value might exceed 1.0 depending on depth.
        # Use optimal threshold search based on accuracy
        best_threshold = np.mean(pred_scores)
        best_acc_temp = 0
        for t in np.linspace(pred_scores.min(), pred_scores.max(), 50):
            y_temp = (pred_scores > t).astype(int)
            acc_temp = accuracy_score(y_true, y_temp)
            if acc_temp > best_acc_temp:
                best_acc_temp = acc_temp
                best_threshold = t
        threshold = best_threshold
        y_pred_binary = (pred_scores > threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred_binary)
        accuracy.append(acc)
        print(f"Accuracy (Threshold={threshold:.4f}): {acc:.4f}")

        # 5. Save Results
        results = pd.DataFrame({
            'True_Label': y_true,
            'SABFS_Score': pred_scores,
            'SABFS_Pred': y_pred_binary
        })
        output_path = f"/home/haoqian/Data/Course/DSAI4204/Project/Predictors/GraphBased/Methodology/Algorithm1/sabfs_multi_root_results_{k_roots}.csv"
        #results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    for k_roots, acc in zip(range(1,top), accuracy):
        print(f"k_roots={k_roots}, accuracy={acc:.4f}")

if __name__ == "__main__":
    main()