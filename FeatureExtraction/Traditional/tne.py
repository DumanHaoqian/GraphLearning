"""
Traditional Feature Engineering Module
======================================

Usage Example:
--------------
    >>> from FeatureExtraction.Traditional.tne import TraditionalFeatureExtraction
    >>> 
    >>> fe = TraditionalFeatureExtraction()
    >>> 
    >>> # 1. PCA dimensionality reduction (adds PCA features to original)
    >>> data = fe.pca_features(data, n_components=5, prefix='pca')
    >>> 
    >>> # 2. KMeans clustering features (adds cluster label + distances)
    >>> data = fe.kmeans_features(data, n_clusters=5, prefix='km')
    >>> 
    >>> # 3. Manual feature combinations
    >>> data = fe.polynomial_features(data, cols=['A', 'B'], degree=2)
    >>> data = fe.interaction_features(data, col_pairs=[('A', 'B'), ('C', 'D')])
    >>> data = fe.ratio_features(data, numerators=['A'], denominators=['B'])
    >>> data = fe.aggregation_features(data, cols=['A', 'B', 'C'])
    >>> 
    >>> # 4. Feature selection
    >>> selected = fe.lasso_selection(X, y, threshold=0.01)
    >>> selected = fe.tree_selection(X, y, threshold=0.01)
    >>> selected = fe.statistical_selection(X, y, method='f_classif', k=10)
    >>> 
    >>> # 5. Get important feature combinations automatically
    >>> data = fe.auto_combine_features(X, y, top_k=5)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.feature_selection import (
    SelectKBest, SelectFromModel, RFE,
    f_classif, f_regression, mutual_info_classif, mutual_info_regression,
    chi2, VarianceThreshold
)
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class TraditionalFeatureExtraction:
    """
    Feature engineering class with PCA, clustering, combinations, and selection.
    
    Attributes:
        pca_model: Fitted PCA model
        kmeans_model: Fitted KMeans model
        scaler: Fitted StandardScaler
        selected_features: List of selected feature names
        feature_importances: Dict of feature importance scores
    """
    
    def __init__(self):
        self.pca_model = None
        self.kmeans_model = None
        self.scaler = None
        self.selected_features = []
        self.feature_importances = {}
        self._fitted_cols = []
    
    # =========================================================================
    # 1. PCA DIMENSIONALITY REDUCTION
    # =========================================================================
    
    def pca_features(self, data: pd.DataFrame, 
                     n_components: Union[int, float] = 5,
                     cols: List[str] = None,
                     prefix: str = 'pca',
                     keep_original: bool = True) -> pd.DataFrame:
        """
        Add PCA-reduced features to the dataset.
        
        Args:
            data: Input DataFrame
            n_components: Number of components (int) or variance ratio (float 0-1)
            cols: Columns to use for PCA (default: all numeric)
            prefix: Prefix for new column names
            keep_original: Whether to keep original columns
            
        Returns:
            DataFrame with PCA features added
        """
        data = data.copy()
        
        # Select numeric columns if not specified
        if cols is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self._fitted_cols = cols
        
        # Standardize before PCA
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data[cols])
        
        # Fit PCA
        self.pca_model = PCA(n_components=n_components)
        pca_result = self.pca_model.fit_transform(scaled_data)
        
        # Create PCA feature names
        pca_cols = [f'{prefix}_{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=data.index)
        
        # Combine with original data
        if keep_original:
            result = pd.concat([data, pca_df], axis=1)
        else:
            non_pca_cols = [c for c in data.columns if c not in cols]
            result = pd.concat([data[non_pca_cols], pca_df], axis=1)
        
        return result
    
    def transform_pca(self, data: pd.DataFrame, prefix: str = 'pca') -> pd.DataFrame:
        """Apply fitted PCA to new data."""
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Call pca_features first.")
        
        data = data.copy()
        scaled_data = self.scaler.transform(data[self._fitted_cols])
        pca_result = self.pca_model.transform(scaled_data)
        
        pca_cols = [f'{prefix}_{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=data.index)
        
        return pd.concat([data, pca_df], axis=1)
    
    # =========================================================================
    # 2. KMEANS CLUSTERING FEATURES
    # =========================================================================
    
    def kmeans_features(self, data: pd.DataFrame,
                        n_clusters: int = 5,
                        cols: List[str] = None,
                        prefix: str = 'km',
                        add_distances: bool = True) -> pd.DataFrame:
        """
        Add KMeans cluster labels and distances as features.
        
        Args:
            data: Input DataFrame
            n_clusters: Number of clusters
            cols: Columns to use for clustering (default: all numeric)
            prefix: Prefix for new column names
            add_distances: Whether to add distance to each cluster center
            
        Returns:
            DataFrame with cluster features added
        """
        data = data.copy()
        
        if cols is None:
            cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Standardize
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[cols])
        
        # Fit KMeans
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(scaled_data)
        
        # Add cluster label
        data[f'{prefix}_cluster'] = cluster_labels
        
        # Add distances to each cluster center
        if add_distances:
            distances = self.kmeans_model.transform(scaled_data)
            for i in range(n_clusters):
                data[f'{prefix}_dist_{i}'] = distances[:, i]
            
            # Add distance to assigned cluster center
            data[f'{prefix}_dist_own'] = np.min(distances, axis=1)
        
        return data
    
    # =========================================================================
    # 3. MANUAL FEATURE COMBINATIONS
    # =========================================================================
    
    def polynomial_features(self, data: pd.DataFrame,
                           cols: List[str],
                           degree: int = 2,
                           interaction_only: bool = False,
                           include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features (A^2, B^2, A*B, etc.).
        
        Args:
            data: Input DataFrame
            cols: Columns to create polynomials from
            degree: Maximum degree of polynomial features
            interaction_only: If True, only interaction terms (no powers)
            include_bias: Whether to include bias column
            
        Returns:
            DataFrame with polynomial features added
        """
        data = data.copy()
        
        poly = PolynomialFeatures(degree=degree, 
                                  interaction_only=interaction_only,
                                  include_bias=include_bias)
        poly_data = poly.fit_transform(data[cols])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(cols)
        
        # Remove original columns from poly output (they're duplicates)
        new_features = []
        new_names = []
        for i, name in enumerate(feature_names):
            if name not in cols:
                new_features.append(poly_data[:, i])
                new_names.append(f'poly_{name}')
        
        if new_features:
            poly_df = pd.DataFrame(
                np.column_stack(new_features),
                columns=new_names,
                index=data.index
            )
            data = pd.concat([data, poly_df], axis=1)
        
        return data
    
    def interaction_features(self, data: pd.DataFrame,
                            col_pairs: List[Tuple[str, str]] = None,
                            cols: List[str] = None) -> pd.DataFrame:
        """
        Create interaction features (A*B).
        
        Args:
            data: Input DataFrame
            col_pairs: List of column pairs to interact, e.g. [('A','B'), ('C','D')]
            cols: If col_pairs not provided, create all pairs from these columns
            
        Returns:
            DataFrame with interaction features added
        """
        data = data.copy()
        
        if col_pairs is None and cols is not None:
            col_pairs = list(combinations(cols, 2))
        elif col_pairs is None:
            return data
        
        for col1, col2 in col_pairs:
            data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
        
        return data
    
    def ratio_features(self, data: pd.DataFrame,
                       numerators: List[str],
                       denominators: List[str],
                       epsilon: float = 1e-8) -> pd.DataFrame:
        """
        Create ratio features (A/B, A/C, etc.).
        
        Args:
            data: Input DataFrame
            numerators: Columns to use as numerators
            denominators: Columns to use as denominators
            epsilon: Small value to avoid division by zero
            
        Returns:
            DataFrame with ratio features added
        """
        data = data.copy()
        
        for num in numerators:
            for denom in denominators:
                if num != denom:
                    data[f'{num}_div_{denom}'] = data[num] / (data[denom] + epsilon)
        
        return data
    
    def aggregation_features(self, data: pd.DataFrame,
                             cols: List[str],
                             aggs: List[str] = None) -> pd.DataFrame:
        """
        Create row-wise aggregation features (sum, mean, std, min, max).
        
        Args:
            data: Input DataFrame
            cols: Columns to aggregate
            aggs: Aggregation functions ['sum', 'mean', 'std', 'min', 'max', 'median']
            
        Returns:
            DataFrame with aggregation features added
        """
        data = data.copy()
        
        if aggs is None:
            aggs = ['sum', 'mean', 'std', 'min', 'max']
        
        prefix = '_'.join(cols[:2]) + '_etc' if len(cols) > 2 else '_'.join(cols)
        
        if 'sum' in aggs:
            data[f'{prefix}_sum'] = data[cols].sum(axis=1)
        if 'mean' in aggs:
            data[f'{prefix}_mean'] = data[cols].mean(axis=1)
        if 'std' in aggs:
            data[f'{prefix}_std'] = data[cols].std(axis=1)
        if 'min' in aggs:
            data[f'{prefix}_min'] = data[cols].min(axis=1)
        if 'max' in aggs:
            data[f'{prefix}_max'] = data[cols].max(axis=1)
        if 'median' in aggs:
            data[f'{prefix}_median'] = data[cols].median(axis=1)
        if 'range' in aggs:
            data[f'{prefix}_range'] = data[cols].max(axis=1) - data[cols].min(axis=1)
        
        return data
    
    def custom_combinations(self, data: pd.DataFrame,
                           formulas: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create custom feature combinations using formulas.
        
        Args:
            data: Input DataFrame
            formulas: List of (name, formula) tuples
                      Formula uses column names, e.g. "A + 0.5*B", "A*B**2", "A*B/C"
                      
        Example:
            >>> formulas = [
            ...     ('custom1', 'A + 0.5*B'),
            ...     ('custom2', 'A * B**2'),
            ...     ('custom3', 'A * B / (C + 0.001)')
            ... ]
            >>> data = fe.custom_combinations(data, formulas)
            
        Returns:
            DataFrame with custom features added
        """
        data = data.copy()
        
        for name, formula in formulas:
            try:
                data[name] = data.eval(formula)
            except Exception as e:
                print(f"Warning: Could not evaluate formula '{formula}': {e}")
        
        return data
    
    def log_features(self, data: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Create log(1+x) features for specified columns."""
        data = data.copy()
        for col in cols:
            min_val = data[col].min()
            if min_val <= 0:
                data[f'{col}_log'] = np.log1p(data[col] - min_val + 1)
            else:
                data[f'{col}_log'] = np.log1p(data[col])
        return data
    
    def binning_features(self, data: pd.DataFrame, 
                        cols: List[str],
                        n_bins: int = 5,
                        strategy: str = 'quantile') -> pd.DataFrame:
        """
        Create binned versions of numerical features.
        
        Args:
            data: Input DataFrame
            cols: Columns to bin
            n_bins: Number of bins
            strategy: 'quantile' or 'uniform'
        """
        data = data.copy()
        
        for col in cols:
            if strategy == 'quantile':
                data[f'{col}_bin'] = pd.qcut(data[col], q=n_bins, labels=False, duplicates='drop')
            else:
                data[f'{col}_bin'] = pd.cut(data[col], bins=n_bins, labels=False)
        
        return data
    
    # =========================================================================
    # 4. FEATURE SELECTION
    # =========================================================================
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series,
                        threshold: float = 0.01,
                        alpha: float = None) -> List[str]:
        """
        Select features using Lasso regularization.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Minimum absolute coefficient to keep feature
            alpha: Lasso alpha (if None, uses LassoCV to find best)
            
        Returns:
            List of selected feature names
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if alpha is None:
            model = LassoCV(cv=5, random_state=42)
        else:
            model = Lasso(alpha=alpha, random_state=42)
        
        model.fit(X_scaled, y)
        
        # Get important features
        importance = np.abs(model.coef_)
        self.feature_importances['lasso'] = dict(zip(X.columns, importance))
        
        selected = X.columns[importance > threshold].tolist()
        self.selected_features = selected
        
        return selected
    
    def tree_selection(self, X: pd.DataFrame, y: pd.Series,
                       threshold: float = 0.01,
                       task: str = 'auto',
                       n_estimators: int = 100) -> List[str]:
        """
        Select features using tree-based importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Minimum importance to keep feature
            task: 'classification', 'regression', or 'auto'
            n_estimators: Number of trees
            
        Returns:
            List of selected feature names
        """
        # Auto-detect task
        if task == 'auto':
            task = 'classification' if y.nunique() <= 10 else 'regression'
        
        if task == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        
        model.fit(X, y)
        
        importance = model.feature_importances_
        self.feature_importances['tree'] = dict(zip(X.columns, importance))
        
        selected = X.columns[importance > threshold].tolist()
        self.selected_features = selected
        
        return selected
    
    def statistical_selection(self, X: pd.DataFrame, y: pd.Series,
                              method: str = 'f_classif',
                              k: int = 10) -> List[str]:
        """
        Select features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: 'f_classif', 'f_regression', 'mutual_info_classif', 
                    'mutual_info_regression', 'chi2'
            k: Number of top features to select
            
        Returns:
            List of selected feature names
        """
        method_map = {
            'f_classif': f_classif,
            'f_regression': f_regression,
            'mutual_info_classif': mutual_info_classif,
            'mutual_info_regression': mutual_info_regression,
            'chi2': chi2
        }
        
        score_func = method_map.get(method, f_classif)
        
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        scores = selector.scores_
        self.feature_importances[method] = dict(zip(X.columns, scores))
        
        selected_mask = selector.get_support()
        selected = X.columns[selected_mask].tolist()
        self.selected_features = selected
        
        return selected
    
    def variance_selection(self, X: pd.DataFrame, 
                          threshold: float = 0.0) -> List[str]:
        """
        Remove features with low variance.
        
        Args:
            X: Feature DataFrame
            threshold: Features with variance <= threshold are removed
            
        Returns:
            List of selected feature names
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_mask = selector.get_support()
        selected = X.columns[selected_mask].tolist()
        self.selected_features = selected
        
        return selected
    
    def rfe_selection(self, X: pd.DataFrame, y: pd.Series,
                      n_features: int = 10,
                      task: str = 'auto') -> List[str]:
        """
        Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            task: 'classification', 'regression', or 'auto'
            
        Returns:
            List of selected feature names
        """
        if task == 'auto':
            task = 'classification' if y.nunique() <= 10 else 'regression'
        
        if task == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        selected_mask = rfe.support_
        selected = X.columns[selected_mask].tolist()
        self.selected_features = selected
        
        return selected
    
    def correlation_selection(self, X: pd.DataFrame, 
                             threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            List of selected feature names (low correlation)
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        selected = [c for c in X.columns if c not in to_drop]
        self.selected_features = selected
        
        return selected
    
    # =========================================================================
    # 5. AUTO FEATURE ENGINEERING
    # =========================================================================
    
    def auto_combine_features(self, X: pd.DataFrame, y: pd.Series,
                              top_k: int = 5,
                              operations: List[str] = None) -> pd.DataFrame:
        """
        Automatically create feature combinations from most important features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            top_k: Number of top features to use for combinations
            operations: List of operations ['multiply', 'add', 'subtract', 'divide', 'power']
            
        Returns:
            DataFrame with new combined features added
        """
        X = X.copy()
        
        if operations is None:
            operations = ['multiply', 'add', 'ratio']
        
        # Get top features by importance
        top_features = self.tree_selection(X, y, threshold=0)[:top_k]
        
        # Create combinations
        for col1, col2 in combinations(top_features, 2):
            if 'multiply' in operations:
                X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            if 'add' in operations:
                X[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
            if 'subtract' in operations:
                X[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
            if 'ratio' in operations:
                X[f'{col1}_div_{col2}'] = X[col1] / (X[col2] + 1e-8)
        
        # Create power features for top features
        if 'power' in operations:
            for col in top_features[:3]:  # Only top 3 for power
                X[f'{col}_squared'] = X[col] ** 2
                X[f'{col}_sqrt'] = np.sqrt(np.abs(X[col]))
        
        return X
    
    def get_feature_importance_df(self, method: str = None) -> pd.DataFrame:
        """
        Get feature importances as a sorted DataFrame.
        
        Args:
            method: 'lasso', 'tree', or specific method name
                    If None, returns all available
                    
        Returns:
            DataFrame with columns ['feature', 'importance', 'method']
        """
        if method and method in self.feature_importances:
            imp = self.feature_importances[method]
            df = pd.DataFrame({
                'feature': list(imp.keys()),
                'importance': list(imp.values()),
                'method': method
            })
            return df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Return all
        all_dfs = []
        for m, imp in self.feature_importances.items():
            df = pd.DataFrame({
                'feature': list(imp.keys()),
                'importance': list(imp.values()),
                'method': m
            })
            all_dfs.append(df)
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True).sort_values(
                ['method', 'importance'], ascending=[True, False]
            )
        return pd.DataFrame()
    
    def select_and_transform(self, X: pd.DataFrame, 
                            features: List[str] = None) -> pd.DataFrame:
        """
        Select only specified features (or previously selected).
        
        Args:
            X: Feature DataFrame
            features: List of feature names (if None, use self.selected_features)
            
        Returns:
            DataFrame with only selected features
        """
        if features is None:
            features = self.selected_features
        
        available = [f for f in features if f in X.columns]
        return X[available].copy()
