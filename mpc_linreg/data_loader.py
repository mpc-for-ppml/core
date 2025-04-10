import csv

def load_party_data(filename):
    """Loads a party's data from a CSV file into X and y."""
    X_local, y_local = [], []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            *features, label = map(float, row)
            features.append(1.0)  # add bias term
            X_local.append(features)
            y_local.append(label)
    return X_local, y_local
