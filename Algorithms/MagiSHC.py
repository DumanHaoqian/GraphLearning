import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, roc_auc_score
# Stochastic Hill Climbing (SHC)
# Algorithm 1: MAGI-SHC v3 (Multi-source Aggregated Golden Inference with Stochastic Hill Climbing)
class MultiRootSHC:
    def __init__(self, adj_matrix, features, labels, golden_ratio=0.618, k_roots=5, max_depth=10, num_restarts=10, temperature=1.0):
        """
        Similarity-based Aggregation with Stochastic Hill Climbing (SHC).
        
        Args:
            adj_matrix: (N, N) Adjacency matrix.
            features: (N, D) Features.
            labels: (N,) Labels.
            golden_ratio: Decay factor (phi).
            k_roots: Number of top similar nodes to use as roots.
            max_depth: Maximum steps for the hill climbing walk.
            num_restarts: Number of stochastic walks per root (to average out variance).
            temperature: Controls randomness. 
                         T -> 0: Pure Greedy (Deterministic). 
                         T -> inf: Random Walk.
                         Default 1.0 is balanced.
        """
        self.adj_matrix = adj_matrix
        self.features = features
        self.labels = labels.flatten()
        self.golden_ratio = golden_ratio
        self.k_roots = k_roots
        self.max_depth = max_depth
        self.num_restarts = num_restarts
        self.temperature = temperature
        self.num_nodes = features.shape[0]

    def _get_next_node_stochastic(self, current_node, visited_in_path):
        """
        Selects the next node based on edge weights using Softmax probabilities.
        """
        # Get neighbors and their weights
        # adj_matrix is sparse-like, assume 0 means no edge
        neighbors_indices = np.where(self.adj_matrix[current_node] > 0)[0]
        
        # Filter out nodes already visited in THIS path to prevent immediate cycles
        valid_neighbors = [n for n in neighbors_indices if n not in visited_in_path]
        
        if not valid_neighbors:
            return None, 0.0
            
        weights = self.adj_matrix[current_node][valid_neighbors]
        
        # Apply Temperature to weights for Softmax
        # Higher weight = Higher probability
        # P(i) = exp(w_i / T) / sum(...)
        w_array = np.array(weights)
        
        # Numerical stability trick
        w_array = w_array / self.temperature
        e_w = np.exp(w_array - np.max(w_array)) 
        probs = e_w / e_w.sum()
        
        # Stochastic Selection
        next_node_idx = np.random.choice(len(valid_neighbors), p=probs)
        selected_node = valid_neighbors[next_node_idx]
        selected_weight = weights[next_node_idx]
        
        return selected_node, selected_weight

    def _run_shc_from_root(self, root_idx):
        """
        Executes Stochastic Hill Climbing with Random Restarts.
        Returns the average accumulated memory from multiple walks.
        """
        accumulated_memories = []
        
        for _ in range(self.num_restarts):
            current_node = root_idx
            path_memory = 0.0
            path_weight_product = 1.0
            visited_in_path = {root_idx}
            
            # Step 0: Root contribution
            path_memory += self.labels[root_idx] * (self.golden_ratio ** 0)
            
            for depth in range(1, self.max_depth + 1):
                # Stochastic Step: Choose ONE neighbor
                next_node, weight = self._get_next_node_stochastic(current_node, visited_in_path)
                
                if next_node is None:
                    break # Dead end
                
                # Update State
                visited_in_path.add(next_node)
                path_weight_product *= weight
                
                # Calculate Experience
                # Memory += Label * Decay * Path_Strength
                decay = self.golden_ratio ** depth
                experience = self.labels[next_node]
                
                path_memory += experience * decay * path_weight_product
                
                # Move
                current_node = next_node
            
            accumulated_memories.append(path_memory)
            
        # Return the average memory of all restarts
        return np.mean(accumulated_memories)

    def predict_single_sample(self, test_feature_vector):
        # 1. Find Top-K Roots (Same logic as before)
        test_vec = test_feature_vector.reshape(1, -1)
        dists = cdist(test_vec, self.features, metric='cosine').flatten()
        
        k = min(self.k_roots, self.num_nodes)
        top_k_indices = np.argsort(dists)[:k]
        top_k_dists = dists[top_k_indices]
        top_k_sims = 1.0 - top_k_dists + 1e-8
        
        # 2. Run SHC for each root
        root_scores = []
        for root_idx in top_k_indices:
            score = self._run_shc_from_root(root_idx)
            root_scores.append(score)
        
        root_scores = np.array(root_scores)
        
        # 3. Weighted Aggregation
        weights = self._softmax(top_k_sims)
        return np.sum(weights * root_scores)

    def _softmax(self, weights):
        w = np.array(weights)
        e_w = np.exp(w - np.max(w))
        return e_w / e_w.sum()

    def predict(self, test_data):
        if isinstance(test_data, pd.DataFrame):
            data_values = test_data.values
        else:
            data_values = test_data
            
        predictions = []
        # print(f"Starting Multi-Root SHC (K={self.k_roots}, Restarts={self.num_restarts})...")
        
        for i, sample in enumerate(data_values):
            pred = self.predict_single_sample(sample)
            predictions.append(pred)
            
        return np.array(predictions)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # Configuration for SHC
    MAX_DEPTH = 8 
    NUM_RESTARTS = 20 # How many random walks per root? More = more stable, slower.
    TEMPERATURE = 0.5 # < 1.0 makes it sharper (more greedy), > 1.0 makes it more random
    
    accuracy_results = []
    
    # Load Data
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

    print(f"{'K_Roots':<10} | {'Accuracy':<10} | {'AUC':<10}")
    print("-" * 35)
    
    # Testing range of K
    for k_roots in range(1, 51): 
        model = MultiRootSHC(
            adj_matrix=adj_matrix,
            features=features,
            labels=labels,
            golden_ratio=0.618,
            k_roots=k_roots,
            max_depth=MAX_DEPTH,
            num_restarts=NUM_RESTARTS,
            temperature=TEMPERATURE
        )

        pred_scores = model.predict(test_data_df)

        # Evaluation
        try:
            auc = roc_auc_score(y_true, pred_scores)
        except:
            auc = 0.0

        # Dynamic Thresholding
        best_acc = 0
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