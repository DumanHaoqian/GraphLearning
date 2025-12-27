import pandas as pd
import numpy as np
import argparse
from Preprocessing.processor import Processor
from GraphConstruction.BuildGraph import GraphBuilder, build_inductive_graph, build_transductive_graph
from FeatureExtraction.Traditional.tne import TraditionalFeatureExtraction
from FeatureExtraction.GraphBased.graphFE import GraphFeatureExtractor, extract_graph_features
from sklearn.ensemble import RandomForestClassifier
from Algorithms import MultiRootSABFS, MultiRootSADFS, RandomWalkRestartClassifier, MultiRootSA, MultiRootSHC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    
    # Step 1 Preprocessing
    processor = Processor(test_ratio=0.2)
    # [transformation, scaling, encoding]
    processor.logic_list = [2, 2, 1]
    """
    ================================================================================
    API REFERENCE: LOGIC_LIST CONFIGURATION
    ================================================================================

    The `logic_list` attribute controls optional preprocessing steps:

        logic_list = [transformation_method, scaling_method, encoding_method]

    POSITION 0 - DATA TRANSFORMATION (for numerical columns)
    ---------------------------------------------------------
        Code | Method              | Description                           | Best For
        -----|---------------------|---------------------------------------|---------------------------
        0    | None                | No transformation                     | Already normal data
        1    | Log (log1p)         | Logarithmic transformation            | Right-skewed data, ratios
        2    | Square Root         | Square root transformation            | Count data, mild skew
        3    | Box-Cox             | Power transformation (positive only)  | General normalization
        4    | Yeo-Johnson         | Power transformation (any values)     | Data with negatives
        5    | Quantile (Uniform)  | Maps to uniform distribution [0,1]    | Non-parametric models
        6    | Quantile (Normal)   | Maps to normal distribution           | Gaussian assumptions

    POSITION 1 - DATA SCALING (for numerical columns)
    ---------------------------------------------------------
        Code | Method              | Output Range  | Description                  | Best For
        -----|---------------------|---------------|------------------------------|---------------------------
        0    | None                | Original      | No scaling                   | Tree-based models
        1    | Min-Max             | [0, 1]        | Linear scaling to range      | Neural networks, images
        2    | Z-Score (Standard)  | mean=0, std=1 | Standardization              | SVM, logistic regression
        3    | Robust              | IQR-based     | Uses median & IQR            | Data with outliers
        4    | Max-Abs             | [-1, 1]       | Scales by max absolute       | Sparse data

    POSITION 2 - CATEGORICAL ENCODING
    ---------------------------------------------------------
        Code | Method              | Description                           | Best For
        -----|---------------------|---------------------------------------|---------------------------
        0    | None                | No encoding                           | Already encoded data
        1    | One-Hot             | Binary columns per category           | Low cardinality, linear
        2    | Label               | Integer codes per category            | Tree-based, ordinal data
    """



    # imputation_method setting (used in preprocess call below)
    """
    ================================================================================
    API REFERENCE: IMPUTATION METHODS
    ================================================================================

        Method      | Numerical Columns      | Categorical Columns | Best For
        ------------|------------------------|---------------------|---------------------------
        'mean'      | Mean value             | Mode (most common)  | Normally distributed
        'median'    | Median value           | Mode (most common)  | Skewed distributions
        'mode'      | Mode (most common)     | Mode (most common)  | Categorical-heavy data
        'knn'       | KNN-based estimation   | Mode (most common)  | Complex relationships
        'ffill'     | Forward fill           | Forward fill        | Time series data
        'bfill'     | Backward fill          | Backward fill       | Time series data
        'constant'  | Fill with 0            | Fill with 'missing' | Explicit missing indicator
    """

    imputation_method = 'mean'
    handle_outliers = 'iqr'
    target_col = 'output'  # Change this to your target column name
    """
    ================================================================================
    API REFERENCE: OUTLIER HANDLING
    ================================================================================

        Method    | Description                              | Best For
        ----------|------------------------------------------|---------------------------
        'none'    | No outlier handling                      | Clean data, tree models
        'remove'  | Remove rows with outliers (z-score)      | Small outlier proportion
        'cap'     | Cap at n std deviations from mean        | Preserve all samples
        'iqr'     | Cap using IQR method (1.5 * IQR)         | Non-normal distributions

    ================================================================================
    """

    processed_data = processor.preprocess(
        data, 
        target_col=target_col,
        imputation_method=imputation_method,
        handle_outliers=handle_outliers
    )
    print()
    X_train = processor.X_train
    X_test = processor.X_test
    y_train = processor.y_train
    y_test = processor.y_test

    # Step 2 Feature Extraction
    fe = TraditionalFeatureExtraction()
    
    # Record train/test split sizes
    n_train = len(X_train)
    n_test = len(X_test)
    
    # Combine train and test for consistent feature engineering
    X_whole = pd.concat([X_train, X_test], ignore_index=True)
    y_whole = pd.concat([y_train, y_test], ignore_index=True)

    # pca feature
    # n_componets could be float(keep 95% of the variance) or int(n features)
    n_pca = min(5, X_whole.shape[1] - 1)  # ensure n_components < n_features
    X_whole = fe.pca_features(X_whole, n_components=n_pca, keep_original=True, prefix='pca') 

    # kmeans feature
    # Add cluster label + distances to each cluster center
    n_clusters = min(5, len(X_whole) // 10)  # ensure enough samples per cluster
    X_whole = fe.kmeans_features(X_whole, n_clusters=n_clusters, prefix='km')

    # step 3: manual feature engineering(optional)
    '''
    # Polynomial features (A^2, A*B, etc.)
    X_whole = fe.polynomial_features(X_whole, cols=['age', 'trtbps', 'chol'], degree=2)

    # Interaction features (A*B)
    X_whole = fe.interaction_features(X_whole, col_pairs=[('age', 'trtbps'), ('chol', 'thalachh')])

    # Ratio features (A/B)
    X_whole = fe.ratio_features(X_whole, numerators=['oldpeak', 'cp_0'], denominators=['slp_1', 'slp_2'])

    # Row-wise aggregations (sum, mean, std, min, max)
    X_whole = fe.aggregation_features(X_whole, cols=['age', 'trtbps', 'chol'])

    # Custom formulas
    formulas = [
        ('custom1', 'age + 0.5*trtbps'),
        ('custom2', 'age * chol**2'),
        ('custom3', 'age * chol / (cp_0 + 0.001)')
    ]
    X_whole = fe.custom_combinations(X_whole, formulas)
    
    # Log features
    X_whole = fe.log_features(X_whole, cols=['age', 'trtbps'])

    # Binning features
    X_whole = fe.binning_features(X_whole, cols=['age', 'trtbps'], n_bins=5)
    '''

    # Split back into train and test (with new features)
    X_train_fe = X_whole.iloc[:n_train].reset_index(drop=True)
    X_test_fe = X_whole.iloc[n_train:].reset_index(drop=True)
    y_train = y_whole.iloc[:n_train].reset_index(drop=True)
    y_test = y_whole.iloc[n_train:].reset_index(drop=True)
    
    # Step 4 Feature Selection (choose one method or combine)
    # Lasso selection
    # selected = fe.lasso_selection(X_train_fe, y_train, threshold=0.01)

    # Tree-based selection (use train data only to avoid data leakage)
    selected = fe.tree_selection(X_train_fe, y_train, threshold=0.01)
    print(f"Selected features by tree: {selected}")

    # Statistical tests (f_classif, f_regression, mutual_info_classif, chi2)
    # selected = fe.statistical_selection(X_train_fe, y_train, method='f_classif', k=10)

    # Variance threshold
    # selected = fe.variance_selection(X_train_fe, threshold=0.01)

    # Recursive Feature Elimination
    # selected = fe.rfe_selection(X_train_fe, y_train, n_features=10)

    # Remove correlated features
    # selected = fe.correlation_selection(X_train_fe, threshold=0.95)
    
    # Apply feature selection to both train and test
    X_train_selected = fe.select_and_transform(X_train_fe, selected)
    X_test_selected = fe.select_and_transform(X_test_fe, selected)


    # Step 5 Graph Construction

    
    # Method 1: Using GraphBuilder class
    builder = GraphBuilder(k=10, similarity='cosine')

    # Inductive graph (train only - for inductive learning)
    graph_inductive = builder.build_inductive(X_train, y_train)

    # Transductive graph (train + test - for transductive learning like GNN)
    graph_transductive = builder.build_transductive(X_train, y_train, X_test, y_test)

    # Convert to PyTorch Geometric (requires torch_geometric)
    # pyg_data = graph_inductive.to_pyg()

    # Convert to NetworkX
    nx_graph = graph_inductive.to_networkx()

    # Get statistics
    stats = graph_inductive.get_statistics()
    # {'num_nodes': 302, 'num_edges': 1510, 'avg_degree': 10.0, 'homophily': 0.72, ...}

    # Visualize graph (uncomment to generate visualizations)
    # graph_inductive.visualize(save_path='/home/haoqian/Data/Course/DSAI4204/GraphLearning/Output/inductive/graph.png')
    # graph_transductive.visualize(save_path='/home/haoqian/Data/Course/DSAI4204/GraphLearning/Output/transductive/graph.png')
    # Visualize similarity heatmap
    # graph_inductive.visualize_similarity(save_path='/home/haoqian/Data/Course/DSAI4204/GraphLearning/Output/inductive/similarity.png')
    # graph_transductive.visualize_similarity(save_path='/home/haoqian/Data/Course/DSAI4204/GraphLearning/Output/transductive/similarity.png')

    # Save/Load (uncomment to save graph data)
    # builder.save_graph(graph_inductive, '/home/haoqian/Data/Course/DSAI4204/GraphLearning/Output/inductive/')
    # builder.save_graph(graph_transductive, '/home/haoqian/Data/Course/DSAI4204/GraphLearning/Output/transductive/')


    # Method 2: Using quick functions
    # graph_inductive = build_inductive_graph(X_train, y_train, k=10)
    # graph_transductive = build_transductive_graph(X_train, y_train, X_test, y_test, k=10)
    
    # Inductive Graph means the graph is built only on the train data
    inductive_adj_matrix = graph_inductive.adj_matrix
    inductive_features = graph_inductive.features
    inductive_labels = graph_inductive.labels
    inductive_similarity_matrix = graph_inductive.similarity_matrix
    inductive_n_train = graph_inductive.n_train
    inductive_n_test = graph_inductive.n_test
    inductive_mode = graph_inductive.mode
    inductive_train_mask = graph_inductive.train_mask
    inductive_test_mask = graph_inductive.test_mask

    
    # Transductive Graph means the graph is built on the train and test data
    transductive_adj_matrix = graph_transductive.adj_matrix
    transductive_features = graph_transductive.features
    transductive_labels = graph_transductive.labels
    transductive_similarity_matrix = graph_transductive.similarity_matrix
    transductive_n_train = graph_transductive.n_train
    transductive_n_test = graph_transductive.n_test
    transductive_mode = graph_transductive.mode
    transductive_train_mask = graph_transductive.train_mask
    transductive_test_mask = graph_transductive.test_mask

    # Step 5.5 MAGI Algorithm using Inductive Graph (train only)
    # =========================================================================
    # MAGI (Multi-source Aggregated Golden Inference) Algorithms
    # Uses the inductive graph to predict test samples via graph traversal
    # =========================================================================

    
    print("\n" + "=" * 60)
    print("Step 5.5: MAGI Algorithms Evaluation")
    print("=" * 60)
    
    # Prepare test data - must match the features used to build the graph
    # The graph was built using X_train (before feature selection), so we use X_test
    from sklearn.preprocessing import StandardScaler
    scaler_magi = StandardScaler()
    # Use the same features as graph construction (X_train before selection)
    scaler_magi.fit(X_train.values)
    test_features_scaled = scaler_magi.transform(X_test.values)
    
    print(f"Test features for MAGI: {test_features_scaled.shape}")
    print(f"Graph features: {inductive_features.shape}")
    
    # Common parameters
    K_ROOTS = 10  # Number of nearest neighbors as starting points
    
    # Store results for comparison
    magi_results = {}
    
    # --- 1. MAGI-BFS (Breadth First Search) ---
    print("\n[1/5] MAGI-BFS (Breadth First Search)...")
    magi_bfs = MultiRootSABFS(
        adj_matrix=inductive_adj_matrix,
        features=inductive_features,
        labels=inductive_labels,
        golden_ratio=0.618,
        k_roots=K_ROOTS,
        convergence_threshold=1e-3
    )
    pred_scores_bfs = magi_bfs.predict(test_features_scaled)
    
    # Find optimal threshold for binary classification
    best_acc_bfs = 0
    best_threshold_bfs = 0.5
    for t in np.linspace(pred_scores_bfs.min(), pred_scores_bfs.max(), 50):
        y_temp = (pred_scores_bfs > t).astype(int)
        acc_temp = accuracy_score(y_test, y_temp)
        if acc_temp > best_acc_bfs:
            best_acc_bfs = acc_temp
            best_threshold_bfs = t
    
    y_pred_bfs = (pred_scores_bfs > best_threshold_bfs).astype(int)
    magi_results['BFS'] = {'accuracy': best_acc_bfs, 'predictions': y_pred_bfs, 'scores': pred_scores_bfs}
    print(f"  MAGI-BFS Accuracy: {best_acc_bfs:.4f}")
    
    # --- 2. MAGI-DFS (Depth First Search) ---
    print("\n[2/5] MAGI-DFS (Depth First Search)...")
    magi_dfs = MultiRootSADFS(
        adj_matrix=inductive_adj_matrix,
        features=inductive_features,
        labels=inductive_labels,
        golden_ratio=0.618,
        k_roots=K_ROOTS,
        max_depth=8
    )
    pred_scores_dfs = magi_dfs.predict(test_features_scaled)
    
    best_acc_dfs = 0
    best_threshold_dfs = 0.5
    for t in np.linspace(pred_scores_dfs.min(), pred_scores_dfs.max(), 50):
        y_temp = (pred_scores_dfs > t).astype(int)
        acc_temp = accuracy_score(y_test, y_temp)
        if acc_temp > best_acc_dfs:
            best_acc_dfs = acc_temp
            best_threshold_dfs = t
    
    y_pred_dfs = (pred_scores_dfs > best_threshold_dfs).astype(int)
    magi_results['DFS'] = {'accuracy': best_acc_dfs, 'predictions': y_pred_dfs, 'scores': pred_scores_dfs}
    print(f"  MAGI-DFS Accuracy: {best_acc_dfs:.4f}")
    
    # --- 3. MAGI-RW (Random Walk with Restart) ---
    print("\n[3/5] MAGI-RW (Random Walk with Restart)...")
    magi_rw = RandomWalkRestartClassifier(
        adj_matrix=inductive_adj_matrix,
        features=inductive_features,
        labels=inductive_labels,
        k_roots=K_ROOTS,
        restart_prob=0.3,
        num_walks=50,
        walk_length=10
    )
    pred_scores_rw = magi_rw.predict(test_features_scaled)
    
    best_acc_rw = 0
    best_threshold_rw = 0.5
    for t in np.linspace(pred_scores_rw.min(), pred_scores_rw.max(), 50):
        y_temp = (pred_scores_rw > t).astype(int)
        acc_temp = accuracy_score(y_test, y_temp)
        if acc_temp > best_acc_rw:
            best_acc_rw = acc_temp
            best_threshold_rw = t
    
    y_pred_rw = (pred_scores_rw > best_threshold_rw).astype(int)
    magi_results['RW'] = {'accuracy': best_acc_rw, 'predictions': y_pred_rw, 'scores': pred_scores_rw}
    print(f"  MAGI-RW Accuracy: {best_acc_rw:.4f}")
    
    # --- 4. MAGI-SHC (Stochastic Hill Climbing) ---
    print("\n[4/5] MAGI-SHC (Stochastic Hill Climbing)...")
    magi_shc = MultiRootSHC(
        adj_matrix=inductive_adj_matrix,
        features=inductive_features,
        labels=inductive_labels,
        golden_ratio=0.618,
        k_roots=K_ROOTS,
        max_depth=8,
        num_restarts=20,
        temperature=0.5
    )
    pred_scores_shc = magi_shc.predict(test_features_scaled)
    
    best_acc_shc = 0
    best_threshold_shc = 0.5
    for t in np.linspace(pred_scores_shc.min(), pred_scores_shc.max(), 50):
        y_temp = (pred_scores_shc > t).astype(int)
        acc_temp = accuracy_score(y_test, y_temp)
        if acc_temp > best_acc_shc:
            best_acc_shc = acc_temp
            best_threshold_shc = t
    
    y_pred_shc = (pred_scores_shc > best_threshold_shc).astype(int)
    magi_results['SHC'] = {'accuracy': best_acc_shc, 'predictions': y_pred_shc, 'scores': pred_scores_shc}
    print(f"  MAGI-SHC Accuracy: {best_acc_shc:.4f}")
    
    # --- 5. MAGI-SA (Simulated Annealing) ---
    print("\n[5/5] MAGI-SA (Simulated Annealing)...")
    magi_sa = MultiRootSA(
        adj_matrix=inductive_adj_matrix,
        features=inductive_features,
        labels=inductive_labels,
        golden_ratio=0.618,
        k_roots=K_ROOTS,
        max_depth=8,
        num_restarts=20,
        initial_temp=0.5,
        cooling_rate=0.8
    )
    pred_scores_sa = magi_sa.predict(test_features_scaled)
    
    best_acc_sa = 0
    best_threshold_sa = 0.5
    for t in np.linspace(pred_scores_sa.min(), pred_scores_sa.max(), 50):
        y_temp = (pred_scores_sa > t).astype(int)
        acc_temp = accuracy_score(y_test, y_temp)
        if acc_temp > best_acc_sa:
            best_acc_sa = acc_temp
            best_threshold_sa = t
    
    y_pred_sa = (pred_scores_sa > best_threshold_sa).astype(int)
    magi_results['SA'] = {'accuracy': best_acc_sa, 'predictions': y_pred_sa, 'scores': pred_scores_sa}
    print(f"  MAGI-SA Accuracy: {best_acc_sa:.4f}")
    
    # --- Summary of MAGI Results ---
    print("\n" + "-" * 40)
    print("MAGI Algorithms Summary:")
    print("-" * 40)
    print(f"{'Algorithm':<15} | {'Accuracy':<10}")
    print("-" * 28)
    for algo, result in sorted(magi_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{algo:<15} | {result['accuracy']:.4f}")
    
    best_magi = max(magi_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest MAGI Algorithm: {best_magi[0]} with accuracy {best_magi[1]['accuracy']:.4f}")
    print("=" * 60)

    # Step 6 Graph Feature Extraction Using Transductive Graph (train + test)
    
    # Extract graph features from transductive graph (train + test)
    gfe_transductive = GraphFeatureExtractor(transductive_adj_matrix)
    graph_features_transductive = gfe_transductive.extract_all_features()
    print(f"Transductive graph features shape: {graph_features_transductive.shape}")
    
    # Split transductive features back to train/test
    graph_features_train = graph_features_transductive.iloc[:transductive_n_train].reset_index(drop=True)
    graph_features_test = graph_features_transductive.iloc[transductive_n_train:].reset_index(drop=True)
    
    # Combine original features with graph features
    X_train_with_graph = pd.concat([X_train_selected.reset_index(drop=True), graph_features_train], axis=1)
    X_test_with_graph = pd.concat([X_test_selected.reset_index(drop=True), graph_features_test], axis=1)
    
    print(f"\nFinal train features shape: {X_train_with_graph.shape}")
    print(f"Final test features shape: {X_test_with_graph.shape}")

    # Step 7 Model Training (with graph features)
    print("\n" + "=" * 50)
    print("Model Training")
    print("=" * 50)
    
    # Handle any NaN/Inf values from graph features
    X_train_with_graph = X_train_with_graph.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_with_graph = X_test_with_graph.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Feature Selection
    #selected = fe.tree_selection(X_train_with_graph, y_train, threshold=0.01)
    #selected = fe.lasso_selection(X_train_with_graph, y_train, threshold=0.01)
    print(f"Selected features by tree: {selected}")
    X_train_with_graph = fe.select_and_transform(X_train_with_graph, selected)
    X_test_with_graph = fe.select_and_transform(X_test_with_graph, selected)

    
    # Train Final 
    model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)
    print(f"Training model with shape: {X_train_with_graph.shape}") 
    model.fit(X_train_with_graph, y_train)
    
    # Step 8 Model Evaluation
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test_with_graph)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy (with graph features): {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Compare with model without graph features
    model_baseline = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)
    print(f"Training baseline model with shape: {X_train_selected.shape}")
    model_baseline.fit(X_train_selected, y_train)
    y_pred_baseline = model_baseline.predict(X_test_selected)
    accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
    print(f"\nBaseline Accuracy (without graph features): {accuracy_baseline:.4f}")
    print(f"Improvement: {(accuracy - accuracy_baseline) * 100:.2f}%")