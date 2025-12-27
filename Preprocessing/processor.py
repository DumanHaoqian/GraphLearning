"""
================================================================================
DATA PREPROCESSING MODULE FOR MACHINE LEARNING
================================================================================

This module provides a comprehensive data preprocessing pipeline for machine 
learning workflows. It handles all common preprocessing tasks including type 
detection, missing value imputation, data cleaning, transformation, scaling, 
encoding, and train-test splitting.

================================================================================
QUICK START EXAMPLE
================================================================================

    >>> from Preprocessing.processor import Processor
    >>> import pandas as pd
    >>> 
    >>> # Load your data
    >>> data = pd.read_csv('your_data.csv')
    >>> 
    >>> # Initialize processor
    >>> processor = Processor(test_ratio=0.2)
    >>> 
    >>> # Configure preprocessing logic: [transformation, scaling, encoding]
    >>> processor.logic_list = [1, 2, 1]  # log transform, z-score, one-hot
    >>> 
    >>> # Run the pipeline
    >>> X_train, X_test, y_train, y_test = processor.preprocess(
    ...     data,
    ...     target_col='target',
    ...     imputation_method='mean',
    ...     handle_outliers='iqr'
    ... )

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

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler,
    PowerTransformer, QuantileTransformer, LabelEncoder
)
from sklearn.model_selection import train_test_split
from scipy import stats
from typing import Tuple, List, Optional, Dict, Union
from enum import IntEnum, Enum


# =============================================================================
# ENUM CLASSES FOR CLEAR METHOD SELECTION
# =============================================================================

class TransformMethod(IntEnum):
    """
    Enumeration for data transformation methods.
    
    Use these constants for clearer code when setting logic_list[0].
    
    Example:
        >>> processor.logic_list[0] = TransformMethod.LOG
    """
    NONE = 0             # No transformation
    LOG = 1              # Log transformation (log1p)
    SQRT = 2             # Square root transformation
    BOX_COX = 3          # Box-Cox transformation
    YEO_JOHNSON = 4      # Yeo-Johnson transformation
    QUANTILE_UNIFORM = 5 # Quantile transform -> uniform
    QUANTILE_NORMAL = 6  # Quantile transform -> normal


class ScaleMethod(IntEnum):
    """
    Enumeration for data scaling methods.
    
    Use these constants for clearer code when setting logic_list[1].
    
    Example:
        >>> processor.logic_list[1] = ScaleMethod.ZSCORE
    """
    NONE = 0      # No scaling
    MINMAX = 1    # Min-Max scaling [0, 1]
    ZSCORE = 2    # Z-score normalization (StandardScaler)
    ROBUST = 3    # Robust scaling (median, IQR)
    MAXABS = 4    # Max-Abs scaling [-1, 1]


class EncodeMethod(IntEnum):
    """
    Enumeration for categorical encoding methods.
    
    Use these constants for clearer code when setting logic_list[2].
    
    Example:
        >>> processor.logic_list[2] = EncodeMethod.ONEHOT
    """
    NONE = 0      # No encoding
    ONEHOT = 1    # One-hot encoding
    LABEL = 2     # Label encoding


class ImputeMethod(str, Enum):
    """
    Enumeration for missing value imputation methods.
    
    Example:
        >>> processor.preprocess(data, imputation_method=ImputeMethod.KNN)
    """
    MEAN = 'mean'         # Mean for numerical, mode for categorical
    MEDIAN = 'median'     # Median for numerical, mode for categorical
    MODE = 'mode'         # Mode for all columns
    KNN = 'knn'           # KNN imputation for numerical
    FFILL = 'ffill'       # Forward fill
    BFILL = 'bfill'       # Backward fill
    CONSTANT = 'constant' # 0 for numerical, 'missing' for categorical


class OutlierMethod(str, Enum):
    """
    Enumeration for outlier handling methods.
    
    Example:
        >>> processor.preprocess(data, handle_outliers=OutlierMethod.IQR)
    """
    NONE = 'none'     # No outlier handling
    REMOVE = 'remove' # Remove outlier rows
    CAP = 'cap'       # Cap at n std deviations
    IQR = 'iqr'       # Cap using IQR method


# =============================================================================
# PROCESSOR CLASS
# =============================================================================

class Processor:
    """
    Comprehensive Data Preprocessing Pipeline for Machine Learning.
    
    This class provides a complete preprocessing workflow with configurable
    steps for handling various data preprocessing tasks. It supports both
    automatic pipeline execution and individual method calls.
    
    Attributes
    ----------
    type_list : list
        Stores the detected type of each variable in order of columns.
        Possible values: 'numerical', 'categorical', 'binary', 'datetime'
        
    test_ratio : float
        Ratio for test set splitting. Default is 0.2 (20% test, 80% train).
        
    logic_list : list[int]
        Controls optional preprocessing steps. Format: [transform, scale, encode]
        - logic_list[0]: Transformation method (0-6, see TransformMethod enum)
        - logic_list[1]: Scaling method (0-4, see ScaleMethod enum)
        - logic_list[2]: Encoding method (0-2, see EncodeMethod enum)
        
    numerical_cols : list[str]
        List of detected numerical column names.
        
    categorical_cols : list[str]
        List of detected categorical column names.
        
    binary_cols : list[str]
        List of detected binary column names (exactly 2 unique values).
        
    datetime_cols : list[str]
        List of detected datetime column names.
        
    scaler : sklearn.preprocessing object or None
        Fitted scaler object for transforming new data.
        
    transformer : sklearn.preprocessing object or None
        Fitted transformer object for transforming new data.
        
    encoders : dict
        Dictionary mapping column names to fitted LabelEncoder objects.
        
    imputation_values : dict
        Dictionary mapping column names to imputation values used.
    
    Methods Summary
    ---------------
    Core Pipeline Methods (called by preprocess()):
        - variable_type_detection() : Detect column types automatically
        - data_imputation()         : Handle missing values
        - data_cleaning()           : Remove duplicates, handle outliers
        - data_transformation()     : Transform numerical distributions
        - data_scaling()            : Scale numerical features
        - data_one_hot_encoding()   : Encode categorical features
        - data_splitting()          : Split into train/test sets
        
    Utility Methods:
        - get_column_types()                : Get detected column types
        - extract_datetime_features()       : Extract date/time components
        - remove_constant_features()        : Remove constant columns
        - remove_high_cardinality_features(): Remove high cardinality categoricals
        - correlation_filter()              : Remove highly correlated features
        - detect_and_convert_dtypes()       : Auto-convert column dtypes
        - get_missing_value_report()        : Generate missing value report
        - get_summary_statistics()          : Generate summary statistics
        - transform_new_data()              : Apply fitted transforms to new data
    
    Examples
    --------
    Basic usage with automatic pipeline:
    
        >>> processor = Processor(test_ratio=0.2)
        >>> processor.logic_list = [
        ...     TransformMethod.LOG,      # Log transformation
        ...     ScaleMethod.ZSCORE,       # Z-score normalization
        ...     EncodeMethod.ONEHOT       # One-hot encoding
        ... ]
        >>> X_train, X_test, y_train, y_test = processor.preprocess(
        ...     data,
        ...     target_col='target',
        ...     imputation_method='mean',
        ...     handle_outliers='iqr'
        ... )
    
    Using individual methods:
    
        >>> processor = Processor()
        >>> processor.variable_type_detection(data)
        >>> data = processor.data_imputation(data, method='knn')
        >>> data = processor.data_cleaning(data, handle_outliers='cap')
        >>> data = processor.data_transformation(data, method=1)
        >>> data = processor.data_scaling(data, method=2)
        >>> data = processor.data_one_hot_encoding(data, method=1)
    
    See Also
    --------
    TransformMethod : Enum for transformation methods
    ScaleMethod     : Enum for scaling methods
    EncodeMethod    : Enum for encoding methods
    ImputeMethod    : Enum for imputation methods
    OutlierMethod   : Enum for outlier handling methods
    """
    
    # =========================================================================
    # CLASS CONSTANTS - Method name mappings for logging
    # =========================================================================
    
    _TRANSFORM_NAMES = {
        0: "none",
        1: "log (log1p)",
        2: "square root",
        3: "box-cox",
        4: "yeo-johnson",
        5: "quantile-uniform",
        6: "quantile-normal"
    }
    
    _SCALE_NAMES = {
        0: "none",
        1: "min-max [0,1]",
        2: "z-score (standard)",
        3: "robust (IQR)",
        4: "max-abs [-1,1]"
    }
    
    _ENCODE_NAMES = {
        0: "none",
        1: "one-hot",
        2: "label"
    }
    
    def __init__(self, test_ratio: float = 0.2):
        """
        Initialize the Processor with default settings.
        
        Parameters
        ----------
        test_ratio : float, default=0.2
            Proportion of data to use for testing (0.0 to 1.0).
            Example: 0.2 means 20% test, 80% train.
            
        Examples
        --------
        >>> processor = Processor(test_ratio=0.3)  # 30% test split
        >>> processor.logic_list = [1, 2, 1]       # Configure preprocessing
        """
        # ---------------------------------------------------------------------
        # Core configuration
        # ---------------------------------------------------------------------
        self.type_list = []       # Detected type per column (in column order)
        self.test_ratio = test_ratio
        self.logic_list = []      # [transform_method, scale_method, encode_method]
        
        # ---------------------------------------------------------------------
        # Fitted transformers (saved for applying to new/test data)
        # These are populated during preprocessing and can be used with
        # transform_new_data() to apply the same transformations to new data.
        # ---------------------------------------------------------------------
        self.scaler = None        # Fitted scaler (MinMaxScaler, StandardScaler, etc.)
        self.transformer = None   # Fitted transformer (PowerTransformer, etc.)
        self.imputers = {}        # Fitted imputers by type {'knn': KNNImputer, ...}
        self.encoders = {}        # Fitted encoders by column {'col_name': LabelEncoder}
        
        # ---------------------------------------------------------------------
        # Column classification (populated by variable_type_detection)
        # ---------------------------------------------------------------------
        self.numerical_cols = []   # Continuous numeric columns
        self.categorical_cols = [] # Discrete category columns (strings or few unique)
        self.binary_cols = []      # Columns with exactly 2 unique values
        self.datetime_cols = []    # Date/time columns
        
        # ---------------------------------------------------------------------
        # Train/test split results (populated by data_splitting)
        # ---------------------------------------------------------------------
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # ---------------------------------------------------------------------
        # Imputation values (stored for applying to new data)
        # Format: {'column_name': imputation_value}
        # ---------------------------------------------------------------------
        self.imputation_values = {}
        
        # ---------------------------------------------------------------------
        # Transformation parameters (stored for reference/debugging)
        # Format: {'column_name': {'method': str, 'shift': float, ...}}
        # ---------------------------------------------------------------------
        self.transform_params = {}
    
    # =========================================================================
    # STEP 1: VARIABLE TYPE DETECTION (Required)
    # =========================================================================
    
    def variable_type_detection(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect the type of each variable in the DataFrame.
        
        This method analyzes each column and classifies it into one of four types:
        - numerical: Continuous numeric variables (int/float with many unique values)
        - categorical: Discrete categories (strings or integers with few unique values)
        - binary: Variables with exactly 2 unique non-null values
        - datetime: Date/time variables (pandas datetime64 type)
        
        Detection Rules
        ---------------
        1. If column is datetime64 type → 'datetime'
        2. If column has exactly 2 unique values → 'binary'
        3. If column is numeric type:
           - If ≤10 unique values AND unique_ratio < 5% → 'categorical'
           - Otherwise → 'numerical'
        4. All other columns → 'categorical'
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to analyze.
            
        Returns
        -------
        dict
            Dictionary with keys 'numerical', 'categorical', 'binary', 'datetime'
            mapping to lists of column names.
            
        Side Effects
        ------------
        Updates the following instance attributes:
        - self.type_list: List of types in column order
        - self.numerical_cols: List of numerical column names
        - self.categorical_cols: List of categorical column names
        - self.binary_cols: List of binary column names
        - self.datetime_cols: List of datetime column names
        
        Examples
        --------
        >>> processor = Processor()
        >>> types = processor.variable_type_detection(df)
        >>> print(types['numerical'])  # ['age', 'income', 'balance']
        >>> print(processor.numerical_cols)  # Same as above
        
        Notes
        -----
        - Binary columns are also suitable for one-hot encoding (creates 1 column)
        - Numeric columns with few unique values may be treated as categorical
          to avoid inappropriate scaling
        - Run this before other preprocessing steps to ensure correct handling
        """
        # Reset all column lists
        self.type_list = []
        self.numerical_cols = []
        self.categorical_cols = []
        self.binary_cols = []
        self.datetime_cols = []
        
        for col in data.columns:
            # -----------------------------------------------------------------
            # Priority 1: Check for datetime type
            # -----------------------------------------------------------------
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                self.datetime_cols.append(col)
                self.type_list.append('datetime')
                
            # -----------------------------------------------------------------
            # Priority 2: Check for binary (exactly 2 unique non-null values)
            # Binary is checked before numeric to catch 0/1 encoded features
            # -----------------------------------------------------------------
            elif data[col].nunique() == 2:
                self.binary_cols.append(col)
                self.type_list.append('binary')
                
            # -----------------------------------------------------------------
            # Priority 3: Check for numerical
            # Numeric columns with very few unique values are treated as
            # categorical to prevent inappropriate transformations
            # -----------------------------------------------------------------
            elif pd.api.types.is_numeric_dtype(data[col]):
                unique_ratio = data[col].nunique() / len(data)
                # Heuristic: ≤10 unique values AND <5% unique ratio → categorical
                if data[col].nunique() <= 10 and unique_ratio < 0.05:
                    self.categorical_cols.append(col)
                    self.type_list.append('categorical')
                else:
                    self.numerical_cols.append(col)
                    self.type_list.append('numerical')
                    
            # -----------------------------------------------------------------
            # Default: Treat as categorical (strings, objects, etc.)
            # -----------------------------------------------------------------
            else:
                self.categorical_cols.append(col)
                self.type_list.append('categorical')
        
        return {
            'numerical': self.numerical_cols.copy(),
            'categorical': self.categorical_cols.copy(),
            'binary': self.binary_cols.copy(),
            'datetime': self.datetime_cols.copy()
        }
    
    # =========================================================================
    # STEP 2: DATA IMPUTATION (Required)
    # =========================================================================
    
    def data_imputation(self, data: pd.DataFrame, method: str = 'mean',
                        knn_neighbors: int = 5) -> pd.DataFrame:
        """
        Handle missing values (NaN/None) in the dataset.
        
        This method fills missing values using various strategies. The choice
        of method should depend on your data characteristics and assumptions.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame with potential missing values.
            
        method : str, default='mean'
            Imputation strategy. Options:
            
            - 'mean'     : Fill numerical with column mean, categorical with mode
                          Best for: Normally distributed numerical data
                          
            - 'median'   : Fill numerical with column median, categorical with mode
                          Best for: Skewed distributions, data with outliers
                          
            - 'mode'     : Fill all columns with mode (most frequent value)
                          Best for: Categorical-heavy datasets
                          
            - 'knn'      : K-Nearest Neighbors imputation for numerical columns,
                          mode for categorical. Uses sklearn.impute.KNNImputer.
                          Best for: Data with complex relationships between features
                          
            - 'ffill'    : Forward fill (propagate last valid value forward)
                          Best for: Time series data
                          
            - 'bfill'    : Backward fill (propagate next valid value backward)
                          Best for: Time series data
                          
            - 'constant' : Fill numerical with 0, categorical with 'missing'
                          Best for: When missing is informative, tree models
                          
        knn_neighbors : int, default=5
            Number of neighbors to use for KNN imputation.
            Only used when method='knn'.
            Higher values → smoother imputations, lower values → more local.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with missing values filled.
            
        Side Effects
        ------------
        - Updates self.imputation_values with values used for each column
        - Updates self.imputers with fitted KNN imputer (if method='knn')
        
        Examples
        --------
        Using mean imputation:
        
            >>> processor = Processor()
            >>> processor.variable_type_detection(data)
            >>> data = processor.data_imputation(data, method='mean')
        
        Using KNN imputation with 10 neighbors:
        
            >>> data = processor.data_imputation(data, method='knn', knn_neighbors=10)
        
        Using enum for clarity:
        
            >>> data = processor.data_imputation(data, method=ImputeMethod.MEDIAN)
        
        Notes
        -----
        - Always run variable_type_detection() before this method
        - KNN imputation can be slow for large datasets
        - ffill/bfill may still leave NaN at edges, so we apply both
        - Imputation values are stored for applying to new/test data
        
        See Also
        --------
        ImputeMethod : Enum class for imputation method constants
        transform_new_data : Apply saved imputation to new data
        get_missing_value_report : Check missing values before imputation
        """
        data = data.copy()
        
        if method == 'knn':
            # KNN imputation for numerical columns
            if self.numerical_cols:
                num_cols_with_nulls = [c for c in self.numerical_cols if data[c].isnull().any()]
                if num_cols_with_nulls:
                    imputer = KNNImputer(n_neighbors=knn_neighbors)
                    data[self.numerical_cols] = imputer.fit_transform(data[self.numerical_cols])
                    self.imputers['knn'] = imputer
            
            # Mode for categorical and binary
            for col in self.categorical_cols + self.binary_cols:
                if data[col].isnull().any():
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else 'missing'
                    self.imputation_values[col] = mode_val
                    data[col].fillna(mode_val, inplace=True)
                    
        elif method == 'mean':
            for col in self.numerical_cols:
                if data[col].isnull().any():
                    mean_val = data[col].mean()
                    self.imputation_values[col] = mean_val
                    data[col].fillna(mean_val, inplace=True)
            for col in self.categorical_cols + self.binary_cols:
                if data[col].isnull().any():
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else 'missing'
                    self.imputation_values[col] = mode_val
                    data[col].fillna(mode_val, inplace=True)
                    
        elif method == 'median':
            for col in self.numerical_cols:
                if data[col].isnull().any():
                    median_val = data[col].median()
                    self.imputation_values[col] = median_val
                    data[col].fillna(median_val, inplace=True)
            for col in self.categorical_cols + self.binary_cols:
                if data[col].isnull().any():
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else 'missing'
                    self.imputation_values[col] = mode_val
                    data[col].fillna(mode_val, inplace=True)
                    
        elif method == 'mode':
            for col in data.columns:
                if data[col].isnull().any():
                    mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else (0 if col in self.numerical_cols else 'missing')
                    self.imputation_values[col] = mode_val
                    data[col].fillna(mode_val, inplace=True)
                    
        elif method == 'ffill':
            data.ffill(inplace=True)
            # If first rows still NaN, use backward fill
            data.bfill(inplace=True)
            
        elif method == 'bfill':
            data.bfill(inplace=True)
            # If last rows still NaN, use forward fill
            data.ffill(inplace=True)
            
        elif method == 'constant':
            for col in self.numerical_cols:
                if data[col].isnull().any():
                    self.imputation_values[col] = 0
                    data[col].fillna(0, inplace=True)
            for col in self.categorical_cols + self.binary_cols:
                if data[col].isnull().any():
                    self.imputation_values[col] = 'missing'
                    data[col].fillna('missing', inplace=True)
        
        return data
    
    # =========================================================================
    # STEP 3: DATA CLEANING (Required)
    # =========================================================================
    
    def data_cleaning(self, data: pd.DataFrame,
                      remove_duplicates: bool = True,
                      handle_outliers: str = 'none',
                      outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        Clean the data by removing duplicates, handling outliers, and fixing formatting.
        
        This method performs several cleaning operations:
        1. Remove duplicate rows (optional)
        2. Strip whitespace from string columns
        3. Handle outliers in numerical columns (optional)
        
        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame to clean.
            
        remove_duplicates : bool, default=True
            Whether to remove duplicate rows.
            - True: Remove all duplicate rows, keeping first occurrence
            - False: Keep all rows including duplicates
            
        handle_outliers : str, default='none'
            Method for handling outliers in numerical columns. Options:
            
            - 'none'   : No outlier handling (keep all values)
                        Best for: Tree-based models, clean data
                        
            - 'remove' : Remove rows where any numerical column has outliers
                        Uses z-score > threshold to identify outliers.
                        Best for: When outliers are errors, small proportions
                        Warning: May significantly reduce dataset size
                        
            - 'cap'    : Cap (clip) values at ±threshold standard deviations
                        Outliers are replaced with boundary values.
                        Best for: When you want to keep all samples
                        
            - 'iqr'    : Cap using Interquartile Range method
                        Lower bound: Q1 - 1.5*IQR, Upper bound: Q3 + 1.5*IQR
                        Best for: Non-normally distributed data
                        
        outlier_threshold : float, default=3.0
            Z-score threshold for 'remove' and 'cap' methods.
            - 2.0: ~95% of normal data retained
            - 3.0: ~99.7% of normal data retained (default)
            - Higher values → fewer outliers detected
            Only used when handle_outliers='remove' or 'cap'.
            
        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame.
            
        Examples
        --------
        Basic cleaning with outlier capping:
        
            >>> data = processor.data_cleaning(
            ...     data,
            ...     remove_duplicates=True,
            ...     handle_outliers='cap',
            ...     outlier_threshold=3.0
            ... )
        
        Using IQR method for outliers:
        
            >>> data = processor.data_cleaning(data, handle_outliers='iqr')
        
        Using enum for clarity:
        
            >>> data = processor.data_cleaning(
            ...     data, 
            ...     handle_outliers=OutlierMethod.IQR
            ... )
        
        Notes
        -----
        - Run after data_imputation() to avoid issues with NaN in z-score calc
        - 'remove' can cause significant data loss if outliers are common
        - 'cap' preserves sample size but may distort extreme values
        - IQR method is more robust to non-normal distributions
        
        See Also
        --------
        OutlierMethod : Enum class for outlier handling method constants
        """
        data = data.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_len = len(data)
            data.drop_duplicates(inplace=True)
            data.reset_index(drop=True, inplace=True)
            removed = initial_len - len(data)
            if removed > 0:
                print(f"Removed {removed} duplicate rows")
        
        # Strip whitespace from string/object columns
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = data[col].str.strip()
                except AttributeError:
                    pass  # Column might have mixed types
        
        # Handle outliers for numerical columns
        if handle_outliers != 'none' and self.numerical_cols:
            for col in self.numerical_cols:
                if col not in data.columns:
                    continue
                    
                if handle_outliers == 'remove':
                    # Calculate z-scores and remove rows with outliers
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        z_scores = np.abs(stats.zscore(col_data))
                        outlier_mask = z_scores > outlier_threshold
                        outlier_indices = col_data.index[outlier_mask]
                        data = data.drop(outlier_indices)
                    
                elif handle_outliers == 'cap':
                    # Cap at threshold standard deviations
                    mean = data[col].mean()
                    std = data[col].std()
                    if std > 0:
                        lower = mean - outlier_threshold * std
                        upper = mean + outlier_threshold * std
                        data[col] = data[col].clip(lower=lower, upper=upper)
                    
                elif handle_outliers == 'iqr':
                    # Use IQR method
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    data[col] = data[col].clip(lower=lower, upper=upper)
            
            data.reset_index(drop=True, inplace=True)
        
        return data
    
    def data_transformation(self, data: pd.DataFrame, method: int = 0) -> pd.DataFrame:
        """
        Transform numerical data to handle skewness and non-normality.
        
        Methods:
        0: No transformation
        1: Log transformation (log1p, handles zeros)
        2: Square root transformation
        3: Box-Cox transformation (requires positive values, auto-shifted)
        4: Yeo-Johnson transformation (handles negative values)
        5: Quantile transformation (uniform distribution)
        6: Quantile transformation (normal/Gaussian distribution)
        
        Args:
            data: Input DataFrame
            method: Transformation method to use
            
        Returns:
            Transformed DataFrame
        """
        if method == 0 or not self.numerical_cols:
            return data
        
        data = data.copy()
        num_cols = [c for c in self.numerical_cols if c in data.columns]
        
        if not num_cols:
            return data
        
        if method == 1:
            # Log transformation (log1p handles zeros)
            for col in num_cols:
                min_val = data[col].min()
                self.transform_params[col] = {'method': 'log', 'shift': 0}
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    self.transform_params[col]['shift'] = shift
                    data[col] = np.log1p(data[col] + shift)
                else:
                    data[col] = np.log1p(data[col])
                    
        elif method == 2:
            # Square root transformation
            for col in num_cols:
                min_val = data[col].min()
                self.transform_params[col] = {'method': 'sqrt', 'shift': 0}
                if min_val < 0:
                    shift = abs(min_val)
                    self.transform_params[col]['shift'] = shift
                    data[col] = np.sqrt(data[col] + shift)
                else:
                    data[col] = np.sqrt(data[col])
                    
        elif method == 3:
            # Box-Cox transformation (requires positive values)
            # Shift data to be positive if needed
            for col in num_cols:
                min_val = data[col].min()
                if min_val <= 0:
                    data[col] = data[col] - min_val + 1
            
            self.transformer = PowerTransformer(method='box-cox', standardize=False)
            data[num_cols] = self.transformer.fit_transform(data[num_cols])
            
        elif method == 4:
            # Yeo-Johnson transformation (handles any values)
            self.transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            data[num_cols] = self.transformer.fit_transform(data[num_cols])
            
        elif method == 5:
            # Quantile transformation to uniform distribution
            self.transformer = QuantileTransformer(
                output_distribution='uniform', 
                random_state=42,
                n_quantiles=min(1000, len(data))
            )
            data[num_cols] = self.transformer.fit_transform(data[num_cols])
            
        elif method == 6:
            # Quantile transformation to normal distribution
            self.transformer = QuantileTransformer(
                output_distribution='normal',
                random_state=42,
                n_quantiles=min(1000, len(data))
            )
            data[num_cols] = self.transformer.fit_transform(data[num_cols])
        
        return data
    
    def data_scaling(self, data: pd.DataFrame, method: int = 0) -> pd.DataFrame:
        """
        Scale numerical data to a standard range.
        
        Methods:
        0: No scaling
        1: Min-Max scaling (scales to [0, 1])
        2: Z-score normalization (StandardScaler, mean=0, std=1)
        3: Robust scaling (uses median and IQR, robust to outliers)
        4: Max-Abs scaling (scales by maximum absolute value, to [-1, 1])
        
        Args:
            data: Input DataFrame
            method: Scaling method to use
            
        Returns:
            Scaled DataFrame
        """
        if method == 0 or not self.numerical_cols:
            return data
        
        data = data.copy()
        num_cols = [c for c in self.numerical_cols if c in data.columns]
        
        if not num_cols:
            return data
        
        if method == 1:
            # Min-Max scaling to [0, 1]
            self.scaler = MinMaxScaler()
            data[num_cols] = self.scaler.fit_transform(data[num_cols])
            
        elif method == 2:
            # Z-score normalization (Standardization)
            self.scaler = StandardScaler()
            data[num_cols] = self.scaler.fit_transform(data[num_cols])
            
        elif method == 3:
            # Robust scaling (median and IQR based)
            self.scaler = RobustScaler()
            data[num_cols] = self.scaler.fit_transform(data[num_cols])
            
        elif method == 4:
            # Max-Abs scaling
            self.scaler = MaxAbsScaler()
            data[num_cols] = self.scaler.fit_transform(data[num_cols])
        
        return data
    
    def data_splitting(self, data: pd.DataFrame, target_col: str = None,
                       stratify: bool = False, 
                       random_state: int = 42) -> Tuple:
        """
        Split data into training and testing sets.
        
        Args:
            data: Input DataFrame
            target_col: Name of the target column (if None, splits entire DataFrame)
            stratify: Whether to use stratified splitting (for classification)
            random_state: Random seed for reproducibility
            
        Returns:
            If target_col is provided: (X_train, X_test, y_train, y_test)
            If target_col is None: (train_data, test_data)
        """
        if target_col is not None and target_col in data.columns:
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            stratify_param = y if stratify else None
            
            try:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, 
                    test_size=self.test_ratio, 
                    random_state=random_state,
                    stratify=stratify_param
                )
            except ValueError as e:
                # Stratification might fail if classes have too few samples
                print(f"Stratification failed: {e}. Falling back to random split.")
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y,
                    test_size=self.test_ratio,
                    random_state=random_state
                )
            
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            train_data, test_data = train_test_split(
                data, 
                test_size=self.test_ratio, 
                random_state=random_state
            )
            return train_data, test_data
    
    def data_one_hot_encoding(self, data: pd.DataFrame, method: int = 0,
                               drop_first: bool = False) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Methods:
        0: No encoding
        1: One-hot encoding (creates binary columns for each category)
        2: Label encoding (converts categories to integers)
        
        Args:
            data: Input DataFrame
            method: Encoding method to use
            drop_first: Whether to drop first category (avoid multicollinearity)
            
        Returns:
            Encoded DataFrame
        """
        if method == 0:
            return data
        
        data = data.copy()
        cols_to_encode = [c for c in (self.categorical_cols + self.binary_cols) 
                         if c in data.columns]
        
        if not cols_to_encode:
            return data
        
        if method == 1:
            # One-hot encoding using pandas get_dummies
            data = pd.get_dummies(data, columns=cols_to_encode, drop_first=drop_first)
            
        elif method == 2:
            # Label encoding
            for col in cols_to_encode:
                le = LabelEncoder()
                # Handle NaN by converting to string first
                data[col] = le.fit_transform(data[col].astype(str))
                self.encoders[col] = le
        
        return data
    
    # ==================== Additional Utility Functions ====================
    
    def get_column_types(self) -> Dict[str, List[str]]:
        """Get the detected column types."""
        return {
            'numerical': self.numerical_cols.copy(),
            'categorical': self.categorical_cols.copy(),
            'binary': self.binary_cols.copy(),
            'datetime': self.datetime_cols.copy()
        }
    
    def extract_datetime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract useful features from datetime columns.
        Creates: year, month, day, day_of_week, hour (if available)
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with datetime columns replaced by extracted features
        """
        data = data.copy()
        
        for col in self.datetime_cols:
            if col not in data.columns:
                continue
            
            # Ensure column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(data[col]):
                try:
                    data[col] = pd.to_datetime(data[col])
                except:
                    continue
            
            data[f'{col}_year'] = data[col].dt.year
            data[f'{col}_month'] = data[col].dt.month
            data[f'{col}_day'] = data[col].dt.day
            data[f'{col}_dayofweek'] = data[col].dt.dayofweek
            
            # Check if time component exists
            if data[col].dt.hour.sum() > 0:
                data[f'{col}_hour'] = data[col].dt.hour
            
            # Drop original datetime column
            data = data.drop(columns=[col])
        
        return data
    
    def remove_constant_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features that have only one unique value (constant).
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with constant features removed
        """
        data = data.copy()
        constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        
        if constant_cols:
            print(f"Removing constant features: {constant_cols}")
            data = data.drop(columns=constant_cols)
            
            # Update column lists
            self.numerical_cols = [c for c in self.numerical_cols if c not in constant_cols]
            self.categorical_cols = [c for c in self.categorical_cols if c not in constant_cols]
            self.binary_cols = [c for c in self.binary_cols if c not in constant_cols]
        
        return data
    
    def remove_high_cardinality_features(self, data: pd.DataFrame,
                                          threshold: float = 0.9) -> pd.DataFrame:
        """
        Remove categorical features with too many unique values.
        Features where unique_values / total_rows > threshold are removed.
        
        Args:
            data: Input DataFrame
            threshold: Maximum allowed ratio of unique values
            
        Returns:
            DataFrame with high cardinality features removed
        """
        data = data.copy()
        high_cardinality_cols = []
        
        for col in self.categorical_cols:
            if col in data.columns:
                cardinality_ratio = data[col].nunique() / len(data)
                if cardinality_ratio > threshold:
                    high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            print(f"Removing high cardinality features: {high_cardinality_cols}")
            data = data.drop(columns=high_cardinality_cols)
            self.categorical_cols = [c for c in self.categorical_cols 
                                     if c not in high_cardinality_cols]
        
        return data
    
    def correlation_filter(self, data: pd.DataFrame,
                           threshold: float = 0.95) -> pd.DataFrame:
        """
        Remove highly correlated numerical features.
        Keeps the first feature of each correlated pair.
        
        Args:
            data: Input DataFrame
            threshold: Correlation threshold above which to remove features
            
        Returns:
            DataFrame with highly correlated features removed
        """
        data = data.copy()
        num_cols = [c for c in self.numerical_cols if c in data.columns]
        
        if not num_cols or len(num_cols) < 2:
            return data
        
        # Calculate correlation matrix
        corr_matrix = data[num_cols].corr().abs()
        
        # Get upper triangle (excluding diagonal)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find columns with correlation above threshold
        to_drop = [column for column in upper_tri.columns 
                   if any(upper_tri[column] > threshold)]
        
        if to_drop:
            print(f"Removing highly correlated features: {to_drop}")
            data = data.drop(columns=to_drop)
            self.numerical_cols = [c for c in self.numerical_cols if c not in to_drop]
        
        return data
    
    def detect_and_convert_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect and convert columns to appropriate dtypes.
        Attempts to convert string columns to numeric or datetime.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with optimized dtypes
        """
        data = data.copy()
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    data[col] = pd.to_numeric(data[col], errors='raise')
                    continue
                except (ValueError, TypeError):
                    pass
                
                # Try to convert to datetime
                try:
                    data[col] = pd.to_datetime(data[col], errors='raise')
                    continue
                except (ValueError, TypeError):
                    pass
        
        return data
    
    def get_missing_value_report(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a report of missing values in the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with missing value statistics for each column
        """
        missing_count = data.isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        
        report = pd.DataFrame({
            'column': data.columns,
            'missing_count': missing_count.values,
            'missing_percentage': missing_pct.values,
            'dtype': data.dtypes.values
        })
        
        report = report[report['missing_count'] > 0].sort_values(
            'missing_percentage', ascending=False
        ).reset_index(drop=True)
        
        return report
    
    def get_summary_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with summary statistics
        """
        # Numerical statistics
        num_stats = data.describe().T
        num_stats['missing'] = data.isnull().sum()
        num_stats['dtype'] = data.dtypes
        
        return num_stats
    
    # ==================== Main Preprocessing Pipeline ====================
    
    def preprocess(self, data: pd.DataFrame, target_col: str = None,
                   imputation_method: str = 'mean',
                   handle_outliers: str = 'none',
                   stratify: bool = False,
                   random_state: int = 42) -> Union[pd.DataFrame, Tuple]:
        """
        Main preprocessing pipeline that executes all preprocessing steps.
        
        The logic_list controls optional preprocessing steps:
        - logic_list[0]: data_transformation method (0=none, 1-6=different methods)
        - logic_list[1]: data_scaling method (0=none, 1-4=different methods)
        - logic_list[2]: data_one_hot_encoding method (0=none, 1-2=different methods)
        
        Example logic_list = [1, 2, 1]:
        - transformation_method = 1 (log transformation)
        - scaling_method = 2 (z-score normalization)
        - encoding_method = 1 (one-hot encoding)
        
        Args:
            data: Input DataFrame
            target_col: Name of target column for supervised learning
            imputation_method: Method for missing value imputation
            handle_outliers: Method for outlier handling ('none', 'remove', 'cap', 'iqr')
            stratify: Whether to use stratified splitting
            random_state: Random seed for reproducibility
            
        Returns:
            If target_col provided: (X_train, X_test, y_train, y_test)
            If target_col is None: (train_data, test_data)
        """
        data = data.copy()
        
        # Set default logic_list if not provided
        if not self.logic_list:
            self.logic_list = [0, 0, 0]
        
        # Ensure logic_list has at least 3 elements
        while len(self.logic_list) < 3:
            self.logic_list.append(0)
        
        # Separate target column before processing (to avoid transforming it)
        # We keep it with the same index so we can rejoin after cleaning
        target_data = None
        if target_col is not None and target_col in data.columns:
            target_data = data[[target_col]].copy()  # Keep as DataFrame to preserve index
            data = data.drop(columns=[target_col])
        
        print("=" * 50)
        print("Starting Preprocessing Pipeline")
        print("=" * 50)
        
        # Step 1: Variable type detection (MUST DO)
        print("\n[Step 1/7] Detecting variable types...")
        type_info = self.variable_type_detection(data)
        print(f"  - Numerical columns: {len(self.numerical_cols)}")
        print(f"  - Categorical columns: {len(self.categorical_cols)}")
        print(f"  - Binary columns: {len(self.binary_cols)}")
        print(f"  - Datetime columns: {len(self.datetime_cols)}")
        
        # Step 2: Data imputation (MUST DO)
        print(f"\n[Step 2/7] Data imputation (method: {imputation_method})...")
        missing_before = data.isnull().sum().sum()
        data = self.data_imputation(data, method=imputation_method)
        missing_after = data.isnull().sum().sum()
        print(f"  - Missing values: {missing_before} -> {missing_after}")
        
        # Step 3: Data cleaning (MUST DO)
        print(f"\n[Step 3/7] Data cleaning (outliers: {handle_outliers})...")
        rows_before = len(data)
        data = self.data_cleaning(data, handle_outliers=handle_outliers)
        rows_after = len(data)
        print(f"  - Rows: {rows_before} -> {rows_after}")
        
        # Step 4: Data transformation (OPTIONAL - controlled by logic_list[0])
        transform_method = self.logic_list[0]
        transform_names = {
            0: "none", 1: "log", 2: "sqrt", 3: "box-cox",
            4: "yeo-johnson", 5: "quantile-uniform", 6: "quantile-normal"
        }
        print(f"\n[Step 4/7] Data transformation (method: {transform_names.get(transform_method, transform_method)})...")
        data = self.data_transformation(data, method=transform_method)
        
        # Step 5: Data scaling (OPTIONAL - controlled by logic_list[1])
        scaling_method = self.logic_list[1]
        scaling_names = {
            0: "none", 1: "min-max", 2: "z-score", 
            3: "robust", 4: "max-abs"
        }
        print(f"\n[Step 5/7] Data scaling (method: {scaling_names.get(scaling_method, scaling_method)})...")
        data = self.data_scaling(data, method=scaling_method)
        
        # Step 6: One-hot encoding (OPTIONAL - controlled by logic_list[2])
        encoding_method = self.logic_list[2]
        encoding_names = {0: "none", 1: "one-hot", 2: "label"}
        print(f"\n[Step 6/7] Encoding (method: {encoding_names.get(encoding_method, encoding_method)})...")
        cols_before = len(data.columns)
        data = self.data_one_hot_encoding(data, method=encoding_method)
        cols_after = len(data.columns)
        print(f"  - Columns: {cols_before} -> {cols_after}")
        
        # Step 7: Data splitting (MUST DO)
        print(f"\n[Step 7/7] Data splitting (test_ratio: {self.test_ratio})...")
        
        # Add target column back if it was separated (match by index after cleaning)
        if target_data is not None:
            # Filter target_data to match remaining rows in data after cleaning
            target_data = target_data.loc[data.index]
            data[target_col] = target_data[target_col].values
            result = self.data_splitting(
                data, target_col=target_col, 
                stratify=stratify, random_state=random_state
            )
            print(f"  - Train samples: {len(result[0])}")
            print(f"  - Test samples: {len(result[1])}")
        else:
            result = self.data_splitting(
                data, target_col=None, 
                stratify=stratify, random_state=random_state
            )
            print(f"  - Train samples: {len(result[0])}")
            print(f"  - Test samples: {len(result[1])}")
        
        print("\n" + "=" * 50)
        print("Preprocessing Complete!")
        print("=" * 50)
        
        return result
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same transformations to new data using fitted transformers.
        Useful for transforming test/validation data after fitting on training data.
        
        Args:
            data: New DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        data = data.copy()
        num_cols = [c for c in self.numerical_cols if c in data.columns]
        
        # Apply imputation
        for col, value in self.imputation_values.items():
            if col in data.columns and data[col].isnull().any():
                data[col].fillna(value, inplace=True)
        
        # Apply transformation
        if self.transformer is not None and num_cols:
            data[num_cols] = self.transformer.transform(data[num_cols])
        
        # Apply scaling
        if self.scaler is not None and num_cols:
            data[num_cols] = self.scaler.transform(data[num_cols])
        
        # Apply encoding
        for col, encoder in self.encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col].astype(str))
        
        return data
