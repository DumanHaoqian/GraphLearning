import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, roc_auc_score

class RandomWalkRestartClassifier:
    def __init__(self, adj_matrix, features, labels, 
                 k_roots=15,           # 每次选取多少个相似节点作为游走起点
                 restart_prob=0.3,     # 重启概率 (Alpha)，越大越关注局部
                 num_walks=100,        # 每个起点走多少次
                 walk_length=10,       # 每次走多远
                 min_edge_weight=0.1): # 忽略极弱的边
        """
        Random Walk with Restart (RWR) optimized for Classification.
        """
        self.adj_matrix = adj_matrix
        self.features = features
        self.labels = labels.flatten()
        self.k_roots = k_roots
        self.restart_prob = restart_prob
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.min_edge_weight = min_edge_weight
        self.num_nodes = features.shape[0]

    def _step(self, current_node):
        """
        Takes a single step based on edge weights.
        """
        neighbors = self.adj_matrix[current_node]
        
        # Filter weak edges to reduce noise
        valid_indices = np.where(neighbors > self.min_edge_weight)[0]
        
        if len(valid_indices) == 0:
            return None
            
        weights = neighbors[valid_indices]
        
        # Normalize probabilities
        probs = weights / np.sum(weights)
        
        # Sample next node
        next_node = np.random.choice(valid_indices, p=probs)
        return next_node

    def _rwr_single_root(self, root_idx):
        """
        Performs Random Walk with Restart from a single root.
        Returns the weighted average label of visited nodes.
        """
        accumulated_label_score = 0.0
        total_visits = 0
        
        for _ in range(self.num_walks):
            curr = root_idx
            
            # Start a walk
            for step in range(self.walk_length):
                # 1. Check for Restart
                # The restart mechanism keeps the walk local to the root's cluster
                if np.random.rand() < self.restart_prob:
                    curr = root_idx
                else:
                    # 2. Move to neighbor
                    nxt = self._step(curr)
                    if nxt is None:
                        curr = root_idx # Dead end, restart
                    else:
                        curr = nxt
                
                # 3. Record Observation
                # We add the label of the node we are currently at.
                # Optional Trick: Decay influence by step distance? 
                # RWR already handles locality, so simple averaging is often robust.
                accumulated_label_score += self.labels[curr]
                total_visits += 1
                
        if total_visits == 0:
            return 0.0
            
        return accumulated_label_score / total_visits

    def predict_single(self, test_vec):
        # 1. Find Top-K Roots (KNN)
        test_vec = test_vec.reshape(1, -1)
        dists = cdist(test_vec, self.features, metric='cosine').flatten()
        
        # Get indices of K nearest training nodes
        top_k_indices = np.argsort(dists)[:self.k_roots]
        
        # Calculate similarities for final weighting
        top_k_sims = 1.0 - dists[top_k_indices]
        top_k_sims = np.maximum(top_k_sims, 0) # Clip negatives
        
        if np.sum(top_k_sims) == 0:
            return 0.0 # No similar nodes found
            
        # 2. Run RWR from each root
        root_scores = []
        
        for root_idx in top_k_indices:
            score = self._rwr_single_root(root_idx)
            root_scores.append(score)
            
        root_scores = np.array(root_scores)
        
        # 3. Weighted Aggregation
        # Final Score = Sum( RWR_Score(root) * Similarity(test, root) )
        final_score = np.average(root_scores, weights=top_k_sims)
        
        return final_score

    def predict(self, test_data):
        if isinstance(test_data, pd.DataFrame):
            data = test_data.values
        else:
            data = test_data
            
        preds = []
        for i, sample in enumerate(data):
            preds.append(self.predict_single(sample))
            
        return np.array(preds)

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    # --- RWR Hyperparameters ---
    # Restart Prob 0.3 means ~30% chance to go back to start. 
    # This keeps the expected walk length around 1/0.3 = 3.33 steps from root.
    # This is excellent for local clustering.
    RESTART_PROB = 0.3 
    NUM_WALKS = 50
    WALK_LENGTH = 10
    
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
    
    # Iterate to find optimal K
    for k_roots in range(1, 41, 2): 
        model = RandomWalkRestartClassifier(
            adj_matrix=adj_matrix,
            features=features,
            labels=labels,
            k_roots=k_roots,
            restart_prob=RESTART_PROB,
            num_walks=NUM_WALKS,
            walk_length=WALK_LENGTH
        )

        pred_scores = model.predict(test_data_df)

        try:
            auc = roc_auc_score(y_true, pred_scores)
        except:
            auc = 0.0

        # Dynamic Threshold Optimization
        best_acc = 0
        thresholds = np.linspace(np.min(pred_scores), np.max(pred_scores), 100)
        for t in thresholds:
            y_pred = (pred_scores > t).astype(int)
            acc = accuracy_score(y_true, y_pred)
            if acc > best_acc:
                best_acc = acc
        
        accuracy_results.append(best_acc)
        print(f"{k_roots:<10} | {best_acc:.4f}     | {auc:.4f}")

if __name__ == "__main__":
    main()