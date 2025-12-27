"""
Graph Construction Module for Machine Learning
===============================================

Builds KNN-based graphs for graph-based machine learning.

Two modes:
1. Inductive: Graph built using only training data (for inductive learning)
2. Transductive: Graph built using train + test data (for transductive learning)

Usage:
------
    >>> from GraphConstruction.BuildGraph import GraphBuilder
    >>> 
    >>> builder = GraphBuilder(k=10, similarity='cosine')
    >>> 
    >>> # Inductive graph (train only)
    >>> graph_inductive = builder.build_inductive(X_train, y_train)
    >>> 
    >>> # Transductive graph (train + test)
    >>> graph_transductive = builder.build_transductive(X_train, y_train, X_test, y_test)
    >>> 
    >>> # Access PyG data
    >>> pyg_data = graph_inductive.to_pyg()
    >>> 
    >>> # Visualize
    >>> graph_inductive.visualize(save_path='graph.png')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel
from sklearn.neighbors import kneighbors_graph
from typing import Optional, Union, Tuple, Dict, List
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Optional PyTorch Geometric imports
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not installed. PyG features disabled.")


@dataclass
class GraphData:
    """Container for graph data and metadata."""
    adj_matrix: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    similarity_matrix: np.ndarray
    n_train: int
    n_test: int
    mode: str  # 'inductive' or 'transductive'
    train_mask: np.ndarray
    test_mask: np.ndarray
    k: int
    
    def to_pyg(self):
        """Convert to PyTorch Geometric Data object."""
        if not HAS_PYG:
            raise ImportError("torch_geometric is required for PyG conversion")
        
        x = torch.tensor(self.features, dtype=torch.float32)
        y = torch.tensor(self.labels, dtype=torch.long)
        
        adj_tensor = torch.tensor(self.adj_matrix, dtype=torch.float32)
        edge_index, edge_weight = dense_to_sparse(adj_tensor)
        
        train_mask = torch.tensor(self.train_mask, dtype=torch.bool)
        test_mask = torch.tensor(self.test_mask, dtype=torch.bool)
        val_mask = torch.zeros(len(self.labels), dtype=torch.bool)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        data.n_train = self.n_train
        data.n_test = self.n_test
        data.mode = self.mode
        
        return data
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph."""
        return nx.from_numpy_array(self.adj_matrix)
    
    def get_statistics(self) -> Dict:
        """Compute graph statistics."""
        n_nodes = self.adj_matrix.shape[0]
        n_edges = int(np.sum(self.adj_matrix > 0) / 2)
        degrees = np.sum(self.adj_matrix > 0, axis=1)
        
        G = self.to_networkx()
        n_components = nx.number_connected_components(G)
        
        try:
            clustering_coef = nx.average_clustering(G, weight='weight')
        except:
            clustering_coef = 0.0
        
        density = nx.density(G)
        
        # Compute homophily (for train nodes only in transductive)
        homophily = self._compute_homophily()
        
        return {
            'num_nodes': n_nodes,
            'num_edges': n_edges,
            'avg_degree': round(np.mean(degrees), 2),
            'num_components': n_components,
            'clustering_coef': round(clustering_coef, 4),
            'density': round(density, 4),
            'homophily': round(homophily, 4),
            'n_train': self.n_train,
            'n_test': self.n_test,
            'mode': self.mode
        }
    
    def _compute_homophily(self) -> float:
        """Compute label homophily ratio."""
        n = self.adj_matrix.shape[0]
        same_label = 0
        total = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.adj_matrix[i, j] > 0:
                    total += 1
                    if self.labels[i] == self.labels[j]:
                        same_label += 1
        
        return same_label / total if total > 0 else 0.0
    
    def visualize(self, save_path: Optional[str] = None, figsize: Tuple = (16, 7)):
        """Visualize graph structure and degree distribution."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        G = self.to_networkx()
        n = self.adj_matrix.shape[0]
        pos = nx.spring_layout(G, seed=42, k=2/np.sqrt(n))
        
        # Draw edges
        ax1 = axes[0]
        edges = G.edges(data=True)
        edge_weights = [d.get('weight', 1) for (u, v, d) in edges]
        if edge_weights:
            nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.3,
                                   width=[w * 2 for w in edge_weights], 
                                   edge_color='lightgray')
        
        # Draw nodes based on mode
        if self.mode == 'inductive':
            self._draw_inductive_nodes(G, pos, ax1)
        else:
            self._draw_transductive_nodes(G, pos, ax1)
        
        ax1.set_title(f'{self.mode.capitalize()} Graph (k={self.k})\n'
                      f'Nodes: {n}, Edges: {int(np.sum(self.adj_matrix > 0) / 2)}')
        ax1.legend(loc='upper left')
        ax1.axis('off')
        
        # Degree distribution
        ax2 = axes[1]
        degrees = np.sum(self.adj_matrix > 0, axis=1)
        
        if self.mode == 'transductive' and self.n_test > 0:
            train_deg = degrees[:self.n_train]
            test_deg = degrees[self.n_train:]
            ax2.hist(train_deg, bins=15, alpha=0.7, color='#2ecc71', 
                     edgecolor='black', label=f'Train (avg: {np.mean(train_deg):.1f})')
            ax2.hist(test_deg, bins=15, alpha=0.5, color='#95a5a6',
                     edgecolor='black', label=f'Test (avg: {np.mean(test_deg):.1f})')
        else:
            ax2.hist(degrees, bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
        
        ax2.axvline(np.mean(degrees), color='red', linestyle='--',
                    label=f'Overall avg: {np.mean(degrees):.1f}')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Count')
        ax2.set_title('Degree Distribution')
        ax2.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Graph saved to: {save_path}")
        plt.show()
        plt.close()
    
    def _draw_inductive_nodes(self, G, pos, ax):
        """Draw nodes for inductive graph (train only)."""
        nodes_0 = [i for i, l in enumerate(self.labels) if l == 0]
        nodes_1 = [i for i, l in enumerate(self.labels) if l == 1]
        
        if nodes_0:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_0, ax=ax,
                                   node_color='#3498db', node_size=80,
                                   alpha=0.8, label=f'Class 0 ({len(nodes_0)})')
        if nodes_1:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_1, ax=ax,
                                   node_color='#e74c3c', node_size=80,
                                   alpha=0.8, label=f'Class 1 ({len(nodes_1)})')
    
    def _draw_transductive_nodes(self, G, pos, ax):
        """Draw nodes for transductive graph (train + test)."""
        train_0 = [i for i in range(self.n_train) if self.labels[i] == 0]
        train_1 = [i for i in range(self.n_train) if self.labels[i] == 1]
        test_nodes = list(range(self.n_train, self.n_train + self.n_test))
        
        if train_0:
            nx.draw_networkx_nodes(G, pos, nodelist=train_0, ax=ax,
                                   node_color='#3498db', node_size=80,
                                   alpha=0.8, label=f'Train Class 0 ({len(train_0)})')
        if train_1:
            nx.draw_networkx_nodes(G, pos, nodelist=train_1, ax=ax,
                                   node_color='#e74c3c', node_size=80,
                                   alpha=0.8, label=f'Train Class 1 ({len(train_1)})')
        if test_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=test_nodes, ax=ax,
                                   node_color='#95a5a6', node_size=80,
                                   alpha=0.8, label=f'Test (Unknown) ({len(test_nodes)})')
    
    def visualize_similarity(self, save_path: Optional[str] = None):
        """Visualize similarity matrix as heatmap."""
        sorted_idx = np.argsort(self.labels)
        sorted_sim = self.similarity_matrix[sorted_idx][:, sorted_idx]
        sorted_labels = self.labels[sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sorted_sim, cmap='RdYlBu_r', aspect='auto')
        
        # Draw class boundary
        changes = np.where(np.diff(sorted_labels) != 0)[0]
        if len(changes) > 0:
            boundary = changes[0] + 1
            ax.axhline(y=boundary - 0.5, color='black', linewidth=2)
            ax.axvline(x=boundary - 0.5, color='black', linewidth=2)
        
        plt.colorbar(im, ax=ax, label='Similarity')
        ax.set_title(f'{self.mode.capitalize()} Similarity Matrix')
        ax.set_xlabel('Node Index (sorted by label)')
        ax.set_ylabel('Node Index (sorted by label)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Similarity heatmap saved to: {save_path}")
        plt.show()
        plt.close()


class GraphBuilder:
    """
    Build KNN graphs for machine learning.
    
    Supports two modes:
    - Inductive: Build graph using training data only
    - Transductive: Build graph using train + test data combined
    
    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbors for KNN graph.
    similarity : str, default='cosine'
        Similarity metric: 'cosine', 'euclidean', 'rbf'
    scale_features : bool, default=True
        Whether to standardize features before building graph.
    symmetric : bool, default=True
        Whether to symmetrize the adjacency matrix.
    """
    
    def __init__(self, k: int = 10, similarity: str = 'cosine',
                 scale_features: bool = True, symmetric: bool = True):
        self.k = k
        self.similarity = similarity
        self.scale_features = scale_features
        self.symmetric = symmetric
        self.scaler = None
    
    def _to_numpy(self, data) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        return np.asarray(data)
    
    def _compute_similarity(self, features: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix."""
        if self.similarity == 'cosine':
            return cosine_similarity(features)
        elif self.similarity == 'euclidean':
            # Convert distance to similarity
            dist = euclidean_distances(features)
            return 1 / (1 + dist)
        elif self.similarity == 'rbf':
            return rbf_kernel(features)
        else:
            raise ValueError(f"Unknown similarity: {self.similarity}")
    
    def _build_knn_adjacency(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Build KNN adjacency matrix from similarity."""
        n = similarity_matrix.shape[0]
        adj = np.zeros((n, n))
        
        for i in range(n):
            sim = similarity_matrix[i].copy()
            sim[i] = -np.inf  # Exclude self-loop
            top_k = np.argsort(sim)[-self.k:]
            adj[i, top_k] = similarity_matrix[i, top_k]
        
        # Symmetrize
        if self.symmetric:
            adj = np.maximum(adj, adj.T)
        
        return adj
    
    def build_inductive(self, X_train, y_train) -> GraphData:
        """
        Build inductive graph using training data only.
        
        In inductive learning, the graph is built solely from training data.
        Test data is handled separately (e.g., by finding neighbors in train graph).
        
        Parameters
        ----------
        X_train : array-like, shape (n_train, n_features)
            Training feature matrix.
        y_train : array-like, shape (n_train,)
            Training labels.
            
        Returns
        -------
        GraphData
            Graph data container with adjacency matrix, features, etc.
        """
        X_train = self._to_numpy(X_train)
        y_train = self._to_numpy(y_train)
        n_train = len(y_train)
        
        # Scale features
        if self.scale_features:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(X_train)
        else:
            features = X_train.copy()
        
        # Compute similarity and build graph
        similarity = self._compute_similarity(features)
        adj_matrix = self._build_knn_adjacency(similarity)
        
        # Create masks (all train for inductive)
        train_mask = np.ones(n_train, dtype=bool)
        test_mask = np.zeros(n_train, dtype=bool)
        
        return GraphData(
            adj_matrix=adj_matrix,
            features=features,
            labels=y_train,
            similarity_matrix=similarity,
            n_train=n_train,
            n_test=0,
            mode='inductive',
            train_mask=train_mask,
            test_mask=test_mask,
            k=self.k
        )
    
    def build_transductive(self, X_train, y_train, X_test, 
                           y_test: Optional = None) -> GraphData:
        """
        Build transductive graph using train + test data.
        
        In transductive learning, the graph includes both train and test nodes.
        Test node labels are masked during training but used for evaluation.
        
        Parameters
        ----------
        X_train : array-like, shape (n_train, n_features)
            Training feature matrix.
        y_train : array-like, shape (n_train,)
            Training labels.
        X_test : array-like, shape (n_test, n_features)
            Test feature matrix.
        y_test : array-like, shape (n_test,), optional
            Test labels (for evaluation, not used in graph construction).
            If None, filled with -1.
            
        Returns
        -------
        GraphData
            Graph data container with train_mask and test_mask.
        """
        X_train = self._to_numpy(X_train)
        y_train = self._to_numpy(y_train)
        X_test = self._to_numpy(X_test)
        
        n_train = len(y_train)
        n_test = len(X_test)
        n_total = n_train + n_test
        
        # Handle test labels
        if y_test is not None:
            y_test = self._to_numpy(y_test)
        else:
            y_test = np.full(n_test, -1)  # Unknown labels
        
        # Combine features
        X_combined = np.vstack([X_train, X_test])
        
        # Scale features
        if self.scale_features:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(X_combined)
        else:
            features = X_combined.copy()
        
        # Compute similarity and build graph
        similarity = self._compute_similarity(features)
        adj_matrix = self._build_knn_adjacency(similarity)
        
        # Combined labels
        labels = np.concatenate([y_train, y_test])
        
        # Create masks
        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        test_mask = np.zeros(n_total, dtype=bool)
        test_mask[n_train:] = True
        
        return GraphData(
            adj_matrix=adj_matrix,
            features=features,
            labels=labels,
            similarity_matrix=similarity,
            n_train=n_train,
            n_test=n_test,
            mode='transductive',
            train_mask=train_mask,
            test_mask=test_mask,
            k=self.k
        )
    
    def save_graph(self, graph_data: GraphData, output_dir: str):
        """Save graph data to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'adj_matrix.npy'), graph_data.adj_matrix)
        np.save(os.path.join(output_dir, 'features.npy'), graph_data.features)
        np.save(os.path.join(output_dir, 'labels.npy'), graph_data.labels)
        np.save(os.path.join(output_dir, 'similarity.npy'), graph_data.similarity_matrix)
        np.save(os.path.join(output_dir, 'train_mask.npy'), graph_data.train_mask)
        np.save(os.path.join(output_dir, 'test_mask.npy'), graph_data.test_mask)
        
        if HAS_PYG:
            pyg_data = graph_data.to_pyg()
            torch.save(pyg_data, os.path.join(output_dir, 'pyg_data.pt'))
        
        # Save metadata
        stats = graph_data.get_statistics()
        with open(os.path.join(output_dir, 'metadata.txt'), 'w') as f:
            for key, val in stats.items():
                f.write(f"{key}: {val}\n")
        
        print(f"Graph saved to: {output_dir}")
    
    def load_graph(self, input_dir: str) -> GraphData:
        """Load graph data from directory."""
        import os
        
        adj_matrix = np.load(os.path.join(input_dir, 'adj_matrix.npy'))
        features = np.load(os.path.join(input_dir, 'features.npy'))
        labels = np.load(os.path.join(input_dir, 'labels.npy'))
        similarity = np.load(os.path.join(input_dir, 'similarity.npy'))
        train_mask = np.load(os.path.join(input_dir, 'train_mask.npy'))
        test_mask = np.load(os.path.join(input_dir, 'test_mask.npy'))
        
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        mode = 'transductive' if n_test > 0 else 'inductive'
        
        return GraphData(
            adj_matrix=adj_matrix,
            features=features,
            labels=labels,
            similarity_matrix=similarity,
            n_train=n_train,
            n_test=n_test,
            mode=mode,
            train_mask=train_mask,
            test_mask=test_mask,
            k=self.k
        )


# Convenience functions
def build_inductive_graph(X_train, y_train, k: int = 10, 
                          similarity: str = 'cosine') -> GraphData:
    """Quick function to build inductive graph."""
    builder = GraphBuilder(k=k, similarity=similarity)
    return builder.build_inductive(X_train, y_train)


def build_transductive_graph(X_train, y_train, X_test, y_test=None,
                              k: int = 10, similarity: str = 'cosine') -> GraphData:
    """Quick function to build transductive graph."""
    builder = GraphBuilder(k=k, similarity=similarity)
    return builder.build_transductive(X_train, y_train, X_test, y_test)

