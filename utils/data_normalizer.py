# utils/data_normalizer.py

import sys

def minmax_normalize(X):
    if not X:
        return X

    num_features = len(X[0])
    for j in range(num_features):
        feature_column = [row[j] for row in X]
        min_val = min(feature_column)
        max_val = max(feature_column)
        range_val = max_val - min_val if max_val != min_val else 1.0

        for i in range(len(X)):
            X[i][j] = (X[i][j] - min_val) / range_val
    
    return X

def zscore_normalize(X):
    if not X:
        return X

    num_features = len(X[0])
    for j in range(num_features):
        feature_column = [row[j] for row in X]
        mean_val = sum(feature_column) / len(feature_column)
        std_val = (sum((x - mean_val) ** 2 for x in feature_column) / len(feature_column)) ** 0.5
        std_val = std_val if std_val != 0 else 1.0

        for i in range(len(X)):
            X[i][j] = (X[i][j] - mean_val) / std_val
    
    return X

def normalize_features(data, method='zscore'):
    normalizers = {
        'minmax': minmax_normalize,
        'zscore': zscore_normalize
    }

    if method not in normalizers:
        raise ValueError(f"Unsupported normalization method: {method}")

    try:
        return normalizers[method](data)
    except ValueError as e:
        print(f"[Normalizer] ❌ Normalization error: {e}")
        sys.exit(1)
