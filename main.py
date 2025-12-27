import pandas as pd
import argparse
from Preprocessing.processor import Processor
from GraphConstruction.BuildGraph import GraphBuilder, build_inductive_graph, build_transductive_graph
from FeatureExtraction.Traditional.tne import TraditionalFeatureExtraction
from sklearn.ensemble import RandomForestClassifier

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

    # Method 2: Quick functions
    graph_inductive = build_inductive_graph(X_train, y_train, k=10)
    graph_transductive = build_transductive_graph(X_train, y_train, X_test, y_test, k=10)
    



    
    # Step 5 Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)
    
    # Step 6 Model Evaluation
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")