# Python/ml/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold

def suggest_missing_strategy(df):
    # Existing code remains unchanged
    missing_counts = df.isnull().sum()
    total_rows = len(df)
    if missing_counts.sum() == 0:
        return 'mean'
    missing_percentages = missing_counts / total_rows
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if any(missing_percentages > 0.5):
        return 'drop'
    if numeric_cols.isin(missing_counts[missing_counts > 0].index).any():
        skewness = df[numeric_cols].skew()
        if any(abs(skewness) > 1):
            return 'median'
        return 'mean'
    if categorical_cols.isin(missing_counts[missing_counts > 0].index).any():
        return 'mode'
    return 'mean'

def target_encode(df, categorical_col, target_col):
    """Perform target encoding on a categorical column using the target variable."""
    target_means = df.groupby(categorical_col)[target_col].mean()
    return df[categorical_col].map(target_means)

def kfold_target_encode(df, categorical_col, target_col, n_splits=5):
    """Perform K-Fold target encoding to prevent data leakage."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    df_encoded = df.copy()
    encoded_col = np.zeros(len(df))
    
    for train_idx, val_idx in kf.split(df):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        target_means = train_df.groupby(categorical_col)[target_col].mean()
        encoded_col[val_idx] = val_df[categorical_col].map(target_means).fillna(train_df[target_col].mean())
    
    df_encoded[categorical_col] = encoded_col
    return df_encoded

def preprocess_data(df, missing_strategy='mean', scaling=True, scaling_columns=None, 
                   encoding='onehot', encoding_columns=None, target_column=None):
    try:
        df_processed = df.copy()
        
        # Handle missing values
        print(f"Handling missing values with strategy: {missing_strategy}")
        if missing_strategy == 'mean':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
        elif missing_strategy == 'median':
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        elif missing_strategy == 'mode':
            for col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        elif missing_strategy == 'drop':
            df_processed.dropna(inplace=True)

        # Identify numeric and categorical columns dynamically
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()

        # Filter columns based on user selection
        if scaling_columns:
            scaling_cols = [col for col in scaling_columns if col in numeric_cols]
        else:
            scaling_cols = numeric_cols if scaling else []
            
        if encoding_columns:
            encoding_cols = [col for col in encoding_columns if col in categorical_cols]
        else:
            encoding_cols = categorical_cols
            
        # Remove target_column from features if provided (for supervised tasks)
        if target_column and target_column in df_processed.columns:
            if target_column in scaling_cols:
                scaling_cols.remove(target_column)
            if target_column in encoding_cols:
                encoding_cols.remove(target_column)

        # Create list of columns that should be passed through without transformation
        passthrough_cols = [col for col in df_processed.columns 
                           if col not in scaling_cols 
                           and col not in encoding_cols 
                           and col != target_column]

        print(f"Scaling columns: {scaling_cols}")
        print(f"Encoding columns: {encoding_cols}")
        print(f"Passthrough columns: {passthrough_cols}")

        # Create preprocessing pipeline
        transformers = []
        if scaling_cols:
            transformers.append(('num', StandardScaler(), scaling_cols))
        
        if encoding_cols:
            if encoding == 'onehot':
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), encoding_cols))
            elif encoding == 'label':
                # Apply Label Encoding to each categorical column
                for col in encoding_cols:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                transformers.append(('cat', 'passthrough', encoding_cols))
            elif encoding == 'target' and target_column and target_column in df_processed.columns:
                # Apply Target Encoding
                for col in encoding_cols:
                    df_processed[col] = target_encode(df_processed, col, target_column)
                transformers.append(('cat', 'passthrough', encoding_cols))
            elif encoding == 'kfold' and target_column and target_column in df_processed.columns:
                # Apply K-Fold Target Encoding
                for col in encoding_cols:
                    df_processed = kfold_target_encode(df_processed, col, target_column)
                transformers.append(('cat', 'passthrough', encoding_cols))
            else:
                # Fallback to onehot if target_column is not provided or invalid encoding
                print(f"Warning: Invalid encoding '{encoding}' or missing target column. Falling back to one-hot encoding.")
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), encoding_cols))

        # Handle columns that should be passed through without transformation
        if passthrough_cols:
            transformers.append(('pass', 'passthrough', passthrough_cols))

        if not transformers:
            print("No columns to preprocess after handling missing values")
            # Ensure target column is preserved if provided
            if target_column and target_column in df.columns:
                return df_processed[[target_column] + [col for col in df_processed.columns if col != target_column]]
            return df_processed

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', force_int_remainder_cols=False)
        transformed_data = preprocessor.fit_transform(df_processed)
        
        # Get feature names after transformation
        feature_names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == 'num' or name == 'pass':
                feature_names.extend(cols)
            elif name == 'cat' and encoding == 'onehot':
                feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out(cols))
            else:
                feature_names.extend(cols)
        
        # If target_column exists, ensure it's included in the output
        if target_column and target_column in df.columns:
            target_data = df_processed[[target_column]].values
            transformed_data = np.hstack((transformed_data, target_data))
            feature_names.append(target_column)

        df_processed = pd.DataFrame(transformed_data, columns=feature_names)
        return df_processed
    except Exception as e:
        print(f"Error in preprocess_data: {str(e)}")
        raise
    
def save_preprocessed_data(df, filename="preprocessed_data.csv"):
    try:
        file_path = f"uploads/{filename}"
        df.to_csv(file_path, index=False)
        return file_path
    except Exception as e:
        print(f"Error saving preprocessed data: {str(e)}")
        raise