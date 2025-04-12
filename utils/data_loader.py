import csv

def load_party_data(filename):
    """Loads a party's data from a CSV file into X and y."""
    X_local, y_local = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            *features, label = map(float, row)
            features.append(1.0)  # add bias term
            X_local.append(features)
            y_local.append(label)
    return X_local, y_local

def load_party_data_adapted(filename):
    """
    Dynamically loads CSV data for a party and returns:
    - user_ids: list of user_id values
    - X_local: list of feature vectors
    - y_local: list of labels (if available, else None)
    - feature_names: names of features (excluding user_id and label)
    - label_name: the name of the label column (if available, else None)
    """
    user_ids = []
    X_local = []
    y_local = []
    feature_names = []
    label_name = None

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        user_id_idx = header.index("user_id")
        label_idx = header.index("purchase_amount") if "purchase_amount" in header else None
        if label_idx is not None:
            label_name = header[label_idx]

        feature_idxs = [i for i in range(len(header)) if i != user_id_idx and i != label_idx]
        feature_names = [header[i] for i in feature_idxs]

        for row in reader:
            user_ids.append(row[user_id_idx])
            X_local.append([float(row[i]) for i in feature_idxs])
            if label_idx is not None:
                y_local.append(float(row[label_idx]))

    return user_ids, X_local, y_local if y_local else None, feature_names, label_name

