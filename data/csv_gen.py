import csv
import random

def generate_csv(filename, n_samples=50, n_features=3, theta=None, noise_std=1.0):
    """Generate synthetic linear data and save to CSV."""
    if theta is None:
        # e.g., for 3 features: [2.0, -1.0, 0.5]
        theta = [random.uniform(-2, 2) for _ in range(n_features)]

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        for _ in range(n_samples):
            features = [random.uniform(0, 10) for _ in range(n_features)]
            target = sum(w * x for w, x in zip(theta, features)) + random.gauss(0, noise_std)
            writer.writerow(features + [target])

# Generate CSVs
generate_csv('data/party1_5.csv', n_samples=50, n_features=5, theta=[1.5, -2.0, -3.0, -6.7, 2.5])
generate_csv('data/party2_5.csv', n_samples=60, n_features=5, theta=[-3.5, 3.0, 3.0, 16.7, 3.5])
generate_csv('data/party3_5.csv', n_samples=200, n_features=5, theta=[4.5, -9.0, 0.0, 3.7, -6.0])
