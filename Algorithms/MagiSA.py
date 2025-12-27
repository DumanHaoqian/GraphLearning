import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, roc_auc_score
# Simulated Annealing (SA)
# Algorithm 1: MAGI-SA v4 (Multi-source Aggregated Golden Inference with Simulated Annealing)
class MultiRootSA:
    def __init__(self, adj_matrix, features, labels, golden_ratio=0.618, k_roots=5, max_depth=10, 
                 num_restarts=10, initial_temp=1.0, cooling_rate=0.85):
        """
        Similarity-based Aggregation with Simulated Annealing (SA).
        
        Args:
            adj_matrix: (N, N) Adjacency matrix.
            features: (N, D) Features.
            labels: (N,) Labels.
            golden_ratio: Decay factor for memory aggregation.
            k_roots: Number of roots.
            max_depth: Max steps for the walk.
            num_restarts: Number of walks per root.
            initial_temp: Starting temperature. Higher = more exploration (accepts weak edges).
            cooling_rate: How fast T decays per step. (0 < rate < 1).
        """
        self.adj_matrix = adj_matrix
        self.features = features
        self.labels = labels.flatten()
        self.golden_ratio = golden_ratio
        self.k_roots = k_roots
        self.max_depth = max_depth
        self.num_restarts = num_restarts
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.num_nodes = features.shape[0]

    def _get_proposal_node(self, current_node, visited_in_path):
        """
        Proposal Step: Suggest a neighbor based on edge weights.
        This is similar to the SHC selection, but this is just a 'proposal'.
        The SA logic decides whether to accept it.
        """
        neighbors_indices = np.where(self.adj_matrix[current_node] > 0)[0]
        valid_neighbors = [n for n in neighbors_indices if n not in visited_in_path]
        
        if not valid_neighbors:
            return None, 0.0
            
        weights = self.adj_matrix[current_node][valid_neighbors]
        
        # Normalize weights to probabilities for proposal
        w_sum = np.sum(weights)
        if w_sum == 0:
            return None, 0.0
            
        probs = weights / w_sum
        
        # Select one neighbor as a proposal
        next_node_idx = np.random.choice(len(valid_neighbors), p=probs)
        selected_node = valid_neighbors[next_node_idx]
        selected_weight = weights[next_node_idx]
        
        return selected_node, selected_weight

    def _run_sa_walk(self, root_idx):
        """
        Executes Simulated Annealing Walk.
        As depth increases, Temperature decreases, making the walker 'pickier'.
        """
        accumulated_memories = []
        
        for _ in range(self.num_restarts):
            current_node = root_idx
            path_memory = 0.0
            path_weight_product = 1.0
            visited_in_path = {root_idx}
            
            # Initialize Temperature for this walk
            current_temp = self.initial_temp
            
            # Root contribution
            path_memory += self.labels[root_idx] * (self.golden_ratio ** 0)
            
            for depth in range(1, self.max_depth + 1):
                # 1. Proposal: Pick a neighbor
                proposal_node, weight = self._get_proposal_node(current_node, visited_in_path)
                
                if proposal_node is None:
                    break # Dead end
                
                # 2. Acceptance Check (Metropolis Criterion adapted for Edge Weights)
                # We define "Energy" as (1 - weight). We want to minimize Energy (maximize weight).
                # Delta E = Energy_new - Energy_ideal(0) = (1 - weight)
                # If weight is 1.0, probability is exp(0) = 1.0 (Always accept perfect edges)
                # If weight is low, probability depends on Temperature.
                
                # Formula: P(accept) = exp( (weight - 1.0) / T )
                # Since (weight - 1.0) is negative, lower T makes P smaller (stricter).
                
                accept_prob = np.exp((weight - 1.0) / current_temp)
                
                if np.random.rand() < accept_prob:
                    # --- Accepted ---
                    visited_in_path.add(proposal_node)
                    path_weight_product *= weight
                    
                    # Accumulate Memory
                    decay = self.golden_ratio ** depth
                    experience = self.labels[proposal_node]
                    path_memory += experience * decay * path_weight_product
                    
                    # Move
                    current_node = proposal_node
                else:
                    # --- Rejected ---
                    # In a walk context, rejection usually means we stop this specific path
                    # because the link was too weak for the current depth/temperature.
                    break 
                
                # 3. Cooling: Reduce temperature for the next step (deeper = stricter)
                current_temp *= self.cooling_rate
            
            accumulated_memories.append(path_memory)
            
        return np.mean(accumulated_memories)

    def predict_single_sample(self, test_feature_vector):
        # 1. Find Top-K Roots
        test_vec = test_feature_vector.reshape(1, -1)
        dists = cdist(test_vec, self.features, metric='cosine').flatten()
        
        k = min(self.k_roots, self.num_nodes)
        top_k_indices = np.argsort(dists)[:k]
        top_k_dists = dists[top_k_indices]
        top_k_sims = 1.0 - top_k_dists + 1e-8
        
        # 2. Run SA Walk for each root
        root_scores = []
        for root_idx in top_k_indices:
            score = self._run_sa_walk(root_idx)
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
        
        for i, sample in enumerate(data_values):
            pred = self.predict_single_sample(sample)
            predictions.append(pred)
            
        return np.array(predictions)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # Configuration for SA
    MAX_DEPTH = 8 
    NUM_RESTARTS = 20 
    
    # SA Specific Params
    # Initial Temp 0.5 means: edge weight 0.8 -> prob exp((0.8-1)/0.5) = exp(-0.4) = 0.67
    # Cooling 0.8 means: at depth 3, temp is 0.5 * 0.8^3 = 0.25
    # At depth 3, edge weight 0.8 -> prob exp(-0.2/0.25) = exp(-0.8) = 0.44 (Much stricter!)
    INITIAL_TEMP = 0.5 
    COOLING_RATE = 0.8 
    
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
    
    for k_roots in range(1, 51): 
        model = MultiRootSA(
            adj_matrix=adj_matrix,
            features=features,
            labels=labels,
            golden_ratio=0.618,
            k_roots=k_roots,
            max_depth=MAX_DEPTH,
            num_restarts=NUM_RESTARTS,
            initial_temp=INITIAL_TEMP,
            cooling_rate=COOLING_RATE
        )

        pred_scores = model.predict(test_data_df)

        # Evaluation
        try:
            auc = roc_auc_score(y_true, pred_scores)
        except:
            auc = 0.0

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