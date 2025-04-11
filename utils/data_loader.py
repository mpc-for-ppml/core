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

def load_party_data_adapted(filename, party_id):
    """
    Load a party's data from a CSV file and extract user_ids, features (X), and optional labels (y).
    Adds a bias term to X.
    
    Assumptions:
    - Org A (party_id == 0) has: user_id, age, income, purchase_amount (label)
    - Org B (party_id == 1) has: user_id, purchase_history (no label)
    - Org C (party_id == 2) has: user_id, web_visits (no label)
    """
    user_ids = []
    X_local = []
    y_local = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            user_id = row[0]
            user_ids.append(user_id)

            if party_id == 0:
                age = float(row[1])
                income = float(row[2])
                label = float(row[3])
                features = [age, income]
                y_local.append(label)
            elif party_id == 1:
                purchase_history = float(row[1])
                features = [purchase_history]
            elif party_id == 2:
                web_visits = float(row[1])
                features = [web_visits]
            else:
                raise ValueError(f"Unsupported party ID: {party_id}")

            features.append(1.0)  # Bias term
            X_local.append(features)

    return user_ids, X_local, y_local if y_local else None
