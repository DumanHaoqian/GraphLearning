"""
Graph-Based Feature Extraction Module
======================================

Extracts node-level features from graph structure for machine learning.

Usage:
------
    >>> from FeatureExtraction.GraphBased.graphFE import GraphFeatureExtractor
    >>> 
    >>> # Initialize with adjacency matrix
    >>> gfe = GraphFeatureExtractor(adj_matrix)
    >>> 
    >>> # Extract all features
    >>> features_df = gfe.extract_all_features()
    >>> 
    >>> # Or extract specific feature groups
    >>> degree_features = gfe.extract_degree_features()
    >>> centrality_features = gfe.extract_centrality_features()
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Union
from scipy import sparse
from scipy.linalg import eigh
import warnings
warnings.filterwarnings('ignore')


class GraphFeatureExtractor:
    """
    Extract graph-based features from adjacency matrix.
    
    Features include:
    - Degree features (degree, weighted degree, centrality)
    - Centrality features (closeness, betweenness, eigenvector, PageRank, etc.)
    - Clustering features (clustering coefficient, triangles, k-core, community)
    - Structural features (eccentricity, shortest paths, neighbor stats)
    - Graphlet features (triangles, wedges, cliques)
    - Spectral features (Laplacian eigenvectors) [NEW]
    - Random Walk features (hitting time, commute time) [NEW]
    - Node2Vec embeddings (optional)
    
    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix (n_nodes x n_nodes), can be weighted.
    """
    
    def __init__(self, adj_matrix: np.ndarray):
        self.adj_matrix = np.array(adj_matrix)
        self.n_nodes = adj_matrix.shape[0]
        
        # Create NetworkX graphs
        self.G = nx.from_numpy_array(self.adj_matrix)
        self.G_unweighted = nx.from_numpy_array((self.adj_matrix > 0).astype(float))
        
        # Store extracted features
        self.all_features = {}
    
    def extract_degree_features(self) -> Dict[str, np.ndarray]:
        """Extract degree-related features."""
        print("  Extracting degree features...")
        features = {}
        
        # 1. Degree
        degrees = dict(self.G.degree())
        features['degree'] = np.array([degrees[i] for i in range(self.n_nodes)])
        
        # 2. Weighted Degree (Strength)
        features['weighted_degree'] = np.sum(self.adj_matrix, axis=1)
        
        # 3. Degree Centrality (normalized degree)
        degree_cent = nx.degree_centrality(self.G)
        features['degree_centrality'] = np.array([degree_cent[i] for i in range(self.n_nodes)])
        
        # 4. Average Neighbor Degree
        avg_neighbor_degree = nx.average_neighbor_degree(self.G, weight='weight')
        features['avg_neighbor_degree'] = np.array([avg_neighbor_degree.get(i, 0) for i in range(self.n_nodes)])
        
        # 5. In/Out degree ratio for directed (here symmetric, so 1)
        features['degree_ratio'] = features['weighted_degree'] / (features['degree'] + 1e-8)
        
        return features
    
    def extract_centrality_features(self) -> Dict[str, np.ndarray]:
        """Extract centrality features."""
        print("  Extracting centrality features...")
        features = {}
        
        # 1. Closeness Centrality
        closeness_cent = nx.closeness_centrality(self.G)
        features['closeness_centrality'] = np.array([closeness_cent[i] for i in range(self.n_nodes)])
        
        # 2. Betweenness Centrality
        betweenness_cent = nx.betweenness_centrality(self.G, weight='weight')
        features['betweenness_centrality'] = np.array([betweenness_cent[i] for i in range(self.n_nodes)])
        
        # 3. Eigenvector Centrality
        try:
            eigenvector_cent = nx.eigenvector_centrality(self.G, max_iter=1000, weight='weight')
            features['eigenvector_centrality'] = np.array([eigenvector_cent[i] for i in range(self.n_nodes)])
        except nx.PowerIterationFailedConvergence:
            print("    Warning: Eigenvector centrality did not converge, using zeros")
            features['eigenvector_centrality'] = np.zeros(self.n_nodes)
        
        # 4. Katz Centrality
        try:
            eigenvalues = np.linalg.eigvals(self.adj_matrix)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            alpha = 0.9 / max_eigenvalue if max_eigenvalue > 0 else 0.1
            katz_cent = nx.katz_centrality(self.G, alpha=alpha, max_iter=1000, weight='weight')
            features['katz_centrality'] = np.array([katz_cent[i] for i in range(self.n_nodes)])
        except:
            print("    Warning: Katz centrality failed, using zeros")
            features['katz_centrality'] = np.zeros(self.n_nodes)
        
        # 5. PageRank
        pagerank = nx.pagerank(self.G, weight='weight')
        features['pagerank'] = np.array([pagerank[i] for i in range(self.n_nodes)])
        
        # 6. HITS (Hub and Authority scores)
        try:
            hubs, authorities = nx.hits(self.G, max_iter=1000)
            features['hub_score'] = np.array([hubs[i] for i in range(self.n_nodes)])
            features['authority_score'] = np.array([authorities[i] for i in range(self.n_nodes)])
        except:
            print("    Warning: HITS failed, using zeros")
            features['hub_score'] = np.zeros(self.n_nodes)
            features['authority_score'] = np.zeros(self.n_nodes)
        
        # 7. Load Centrality
        try:
            load_cent = nx.load_centrality(self.G, weight='weight')
            features['load_centrality'] = np.array([load_cent[i] for i in range(self.n_nodes)])
        except:
            features['load_centrality'] = np.zeros(self.n_nodes)
        
        return features
    
    def extract_clustering_features(self) -> Dict[str, np.ndarray]:
        """Extract clustering and community features."""
        print("  Extracting clustering features...")
        features = {}
        
        # 1. Local Clustering Coefficient
        clustering_coef = nx.clustering(self.G, weight='weight')
        features['clustering_coefficient'] = np.array([clustering_coef[i] for i in range(self.n_nodes)])
        
        # 2. Triangle Count
        triangles = nx.triangles(self.G_unweighted)
        features['triangle_count'] = np.array([triangles[i] for i in range(self.n_nodes)])
        
        # 3. Square Clustering
        features['square_clustering'] = np.array([
            nx.square_clustering(self.G_unweighted, [i])[i] for i in range(self.n_nodes)
        ])
        
        # 4. K-Core Number
        core_numbers = nx.core_number(self.G_unweighted)
        features['k_core'] = np.array([core_numbers[i] for i in range(self.n_nodes)])
        
        # 5. Community Detection (Label Propagation)
        try:
            communities = list(nx.community.label_propagation_communities(self.G_unweighted))
            community_map = {}
            for idx, community in enumerate(communities):
                for node in community:
                    community_map[node] = idx
            features['community_id'] = np.array([community_map[i] for i in range(self.n_nodes)])
            features['community_size'] = np.array([
                len([n for n in community_map if community_map[n] == community_map[i]])
                for i in range(self.n_nodes)
            ])
        except:
            print("    Warning: Community detection failed")
            features['community_id'] = np.zeros(self.n_nodes)
            features['community_size'] = np.zeros(self.n_nodes)
        
        # 6. Louvain Community Detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.G_unweighted)
            features['louvain_community'] = np.array([partition[i] for i in range(self.n_nodes)])
            louvain_sizes = {}
            for node, comm in partition.items():
                if comm not in louvain_sizes:
                    louvain_sizes[comm] = 0
                louvain_sizes[comm] += 1
            features['louvain_community_size'] = np.array([louvain_sizes[partition[i]] for i in range(self.n_nodes)])
        except ImportError:
            print("    Note: python-louvain not installed, skipping Louvain community")
        except:
            pass
        
        return features
    
    def extract_structural_features(self) -> Dict[str, np.ndarray]:
        """Extract structural role features."""
        print("  Extracting structural features...")
        features = {}
        
        # 1. Eccentricity
        if nx.is_connected(self.G_unweighted):
            eccentricity = nx.eccentricity(self.G_unweighted)
            features['eccentricity'] = np.array([eccentricity[i] for i in range(self.n_nodes)])
        else:
            features['eccentricity'] = np.zeros(self.n_nodes)
            largest_cc = max(nx.connected_components(self.G_unweighted), key=len)
            subgraph = self.G_unweighted.subgraph(largest_cc).copy()
            ecc = nx.eccentricity(subgraph)
            for node in largest_cc:
                features['eccentricity'][node] = ecc[node]
        
        # 2. Average Shortest Path Length
        features['avg_shortest_path'] = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            lengths = nx.single_source_shortest_path_length(self.G_unweighted, i)
            if len(lengths) > 1:
                features['avg_shortest_path'][i] = np.mean([l for n, l in lengths.items() if n != i])
        
        # 3. 2-hop Neighbors Count
        features['neighbors_2hop'] = np.array([
            len(set(nx.single_source_shortest_path_length(self.G_unweighted, i, cutoff=2).keys()) - {i})
            for i in range(self.n_nodes)
        ])
        
        # 4. Neighbor Weight Statistics
        neighbor_weight_stats = []
        for i in range(self.n_nodes):
            neighbor_weights = self.adj_matrix[i][self.adj_matrix[i] > 0]
            if len(neighbor_weights) > 0:
                neighbor_weight_stats.append([
                    np.mean(neighbor_weights),
                    np.std(neighbor_weights),
                    np.max(neighbor_weights),
                    np.min(neighbor_weights)
                ])
            else:
                neighbor_weight_stats.append([0, 0, 0, 0])
        
        neighbor_weight_stats = np.array(neighbor_weight_stats)
        features['neighbor_weight_mean'] = neighbor_weight_stats[:, 0]
        features['neighbor_weight_std'] = neighbor_weight_stats[:, 1]
        features['neighbor_weight_max'] = neighbor_weight_stats[:, 2]
        features['neighbor_weight_min'] = neighbor_weight_stats[:, 3]
        
        return features
    
    def extract_graphlet_features(self) -> Dict[str, np.ndarray]:
        """Extract graphlet-related features."""
        print("  Extracting graphlet features...")
        features = {}
        
        degrees = np.array([self.G_unweighted.degree(i) for i in range(self.n_nodes)])
        triangles = np.array([nx.triangles(self.G_unweighted, i) for i in range(self.n_nodes)])
        
        # Open triangles (wedges)
        wedges = degrees * (degrees - 1) / 2
        features['open_triangles'] = wedges - triangles
        features['wedge_count'] = wedges
        
        # Closure ratio (triangles / wedges)
        features['closure_ratio'] = np.divide(triangles, wedges, 
                                               out=np.zeros_like(triangles, dtype=float), 
                                               where=wedges > 0)
        
        # 4-clique participation
        features['clique_4'] = np.zeros(self.n_nodes)
        try:
            cliques = list(nx.enumerate_all_cliques(self.G_unweighted))
            for clique in cliques:
                if len(clique) == 4:
                    for node in clique:
                        features['clique_4'][node] += 1
                elif len(clique) > 4:
                    break  # Stop early for efficiency
        except:
            pass
        
        return features
    
    # =========================================================================
    # NEW FEATURE 1: Spectral Features (Laplacian-based)
    # =========================================================================
    
    def extract_spectral_features(self, n_components: int = 8) -> Dict[str, np.ndarray]:
        """
        Extract spectral features based on graph Laplacian.
        
        Spectral features capture global graph structure through eigenvectors
        of the normalized Laplacian matrix.
        
        Parameters
        ----------
        n_components : int
            Number of eigenvector components to extract.
        """
        print("  Extracting spectral features...")
        features = {}
        
        # Compute degree matrix
        degrees = np.sum(self.adj_matrix, axis=1)
        D = np.diag(degrees)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-8))
        
        # Normalized Laplacian: L_norm = I - D^(-1/2) * A * D^(-1/2)
        L_norm = np.eye(self.n_nodes) - D_inv_sqrt @ self.adj_matrix @ D_inv_sqrt
        
        # Compute eigenvalues and eigenvectors
        n_comp = min(n_components, self.n_nodes - 1)
        try:
            eigenvalues, eigenvectors = eigh(L_norm)
            
            # Skip first eigenvector (constant for connected graph)
            # Use eigenvectors corresponding to smallest non-zero eigenvalues
            for i in range(1, n_comp + 1):
                if i < len(eigenvalues):
                    features[f'spectral_{i}'] = eigenvectors[:, i]
            
            # Fiedler value (second smallest eigenvalue) - algebraic connectivity
            if len(eigenvalues) > 1:
                features['fiedler_value'] = np.full(self.n_nodes, eigenvalues[1])
                features['fiedler_vector'] = eigenvectors[:, 1]
            
            # Spectral gap
            if len(eigenvalues) > 2:
                features['spectral_gap'] = np.full(self.n_nodes, eigenvalues[2] - eigenvalues[1])
            
        except Exception as e:
            print(f"    Warning: Spectral decomposition failed: {e}")
            for i in range(1, n_comp + 1):
                features[f'spectral_{i}'] = np.zeros(self.n_nodes)
        
        return features
    
    # =========================================================================
    # NEW FEATURE 2: Random Walk Features
    # =========================================================================
    
    def extract_random_walk_features(self, n_steps: int = 10) -> Dict[str, np.ndarray]:
        """
        Extract random walk-based features.
        
        Random walk features capture local neighborhood structure through
        transition probabilities and stationary distributions.
        
        Parameters
        ----------
        n_steps : int
            Number of steps for random walk analysis.
        """
        print("  Extracting random walk features...")
        features = {}
        
        # Compute transition matrix P = D^(-1) * A
        degrees = np.sum(self.adj_matrix, axis=1)
        D_inv = np.diag(1.0 / (degrees + 1e-8))
        P = D_inv @ self.adj_matrix
        
        # 1. Stationary distribution (proportional to degree for undirected)
        stationary = degrees / (np.sum(degrees) + 1e-8)
        features['stationary_prob'] = stationary
        
        # 2. Return probability after n steps (diagonal of P^n)
        P_n = np.linalg.matrix_power(P, n_steps)
        features['return_prob'] = np.diag(P_n)
        
        # 3. Self-loop probability (for random walk with restart)
        restart_prob = 0.15
        P_rwr = (1 - restart_prob) * P + restart_prob * np.eye(self.n_nodes)
        P_rwr_n = np.linalg.matrix_power(P_rwr, n_steps)
        features['rwr_return_prob'] = np.diag(P_rwr_n)
        
        # 4. Average transition probability to neighbors
        features['avg_transition_prob'] = np.array([
            np.mean(P[i][self.adj_matrix[i] > 0]) if degrees[i] > 0 else 0
            for i in range(self.n_nodes)
        ])
        
        # 5. Personalized PageRank scores (PPR)
        try:
            ppr_scores = np.zeros(self.n_nodes)
            for i in range(min(self.n_nodes, 100)):  # Sample nodes for efficiency
                personalization = {j: 1.0 if j == i else 0.0 for j in range(self.n_nodes)}
                ppr = nx.pagerank(self.G, personalization=personalization, weight='weight')
                ppr_scores[i] = ppr[i]
            
            # For remaining nodes, use average
            if self.n_nodes > 100:
                avg_ppr = np.mean(ppr_scores[:100])
                ppr_scores[100:] = avg_ppr
            
            features['ppr_self'] = ppr_scores
        except:
            features['ppr_self'] = np.zeros(self.n_nodes)
        
        # 6. Hitting time estimate (average steps to reach node from random start)
        # Approximated using inverse of stationary distribution
        features['hitting_time_approx'] = 1.0 / (stationary + 1e-8)
        
        return features
    
    def extract_node2vec_features(self, dimensions: int = 64, walk_length: int = 30,
                                   num_walks: int = 100, p: float = 1, q: float = 1) -> Optional[Dict[str, np.ndarray]]:
        """Extract Node2Vec embedding features."""
        print("  Extracting Node2Vec features...")
        
        try:
            from node2vec import Node2Vec
            
            node2vec = Node2Vec(
                self.G_unweighted,
                dimensions=dimensions,
                walk_length=walk_length,
                num_walks=num_walks,
                p=p,
                q=q,
                workers=4,
                quiet=True
            )
            
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            embeddings = np.zeros((self.n_nodes, dimensions))
            for i in range(self.n_nodes):
                if str(i) in model.wv:
                    embeddings[i] = model.wv[str(i)]
            
            features = {f'node2vec_{i}': embeddings[:, i] for i in range(dimensions)}
            return features
            
        except ImportError:
            print("    Warning: node2vec package not installed.")
            print("    Install with: pip install node2vec")
            return None
        except Exception as e:
            print(f"    Warning: Node2Vec failed: {e}")
            return None
    
    def extract_all_features(self, allow_list: List[bool] = [True, True, True, True, True, True, True, True], 
    node2vec_dim: int = 32) -> pd.DataFrame:
        
        print("\n" + "=" * 50)
        print("Extracting Graph Features")
        print("=" * 50)
        
        all_features = {}
        
        # Core features
        all_features.update(self.extract_degree_features())
        all_features.update(self.extract_centrality_features())
        all_features.update(self.extract_clustering_features())
        all_features.update(self.extract_structural_features())
        all_features.update(self.extract_graphlet_features())
        
        # New features
        all_features.update(self.extract_spectral_features())
        all_features.update(self.extract_random_walk_features())
        
        if allow_list[7]:
            n2v_features = self.extract_node2vec_features(dimensions=node2vec_dim)
            if n2v_features:
                all_features.update(n2v_features)
        
        # Store and convert to DataFrame
        self.all_features = all_features
        df = pd.DataFrame(all_features)
        
        print(f"\nExtracted {len(df.columns)} features for {len(df)} nodes")
        print("=" * 50)
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of extracted feature names."""
        return list(self.all_features.keys())
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary statistics of extracted features."""
        if not self.all_features:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.all_features)
        return df.describe().T


# Convenience function
def extract_graph_features(adj_matrix: np.ndarray, 
                           include_node2vec: bool = False) -> pd.DataFrame:
    """Quick function to extract all graph features."""
    extractor = GraphFeatureExtractor(adj_matrix)
    return extractor.extract_all_features()
