import csv
import random
import os
import argparse
import json
from pathlib import Path
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def generate_logreg_csv(filename, n_samples, n_features, theta, noise_std):
    if theta is None:
        theta = [random.uniform(-2, 2) for _ in range(n_features)]
    elif len(theta) != n_features:
        raise ValueError(f"Length of theta ({len(theta)}) must match number of features ({n_features})")

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Write header row: x0, x1, ..., xn, y
        header = [f"x{i}" for i in range(n_features)] + ["y"]
        writer.writerow(header)

        for _ in range(n_samples):
            features = [random.uniform(0, 10) for _ in range(n_features)]
            linear_combination = sum(w * x for w, x in zip(theta, features))
            linear_combination += random.gauss(0, noise_std)  # add noise
            probability = sigmoid(linear_combination)
            label = 1 if random.random() < probability else 0
            writer.writerow(features + [label])

    print(f"✅ Generated {filename} with {n_samples} samples, {n_features} features.")

def get_next_session_folder(base_folder="data"):
    Path(base_folder).mkdir(exist_ok=True)
    existing_sessions = [f for f in os.listdir(base_folder) if f.startswith("session_")]
    session_ids = [int(f.split("_")[1]) for f in existing_sessions if f.split("_")[1].isdigit()]
    next_id = max(session_ids, default=0) + 1
    session_folder = os.path.join(base_folder, f"session_{str(next_id).zfill(2)}")
    os.makedirs(session_folder)
    return session_folder

def main():
    parser = argparse.ArgumentParser(description="Generate multi-party synthetic datasets for logistic regression.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--folder", type=str, help="Optional custom folder name for output (inside ./data)")
    args = parser.parse_args()

    if args.folder:
        session_folder = os.path.join("data", args.folder)
        os.makedirs(session_folder, exist_ok=True)
    else:
        session_folder = get_next_session_folder()

    print(f"📁 Generating dataset in folder: {session_folder}\n")

    with open(args.config, 'r') as f:
        config = json.load(f)

    for party in config["parties"]:
        filename = os.path.join(session_folder, f"{party['name']}.csv")
        generate_logreg_csv(
            filename=filename,
            n_samples=party["samples"],
            n_features=party["features"],
            theta=party.get("theta"),
            noise_std=party.get("noise", 1.0)
        )

if __name__ == "__main__":
    main()
