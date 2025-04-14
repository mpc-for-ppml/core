import csv
import uuid
import random
import os
import json
import argparse

# Config for generation
CONFIG = {
    "total_users": 500,
    "min_shared_users": 20,
    "orgs": {
        "A": ["user_id", "age", "income", "will_purchase"],
        "B": ["user_id", "purchase_history"],
        "C": ["user_id", "web_visits"]
    }
}

# Generate unique user_ids and assign them to subsets
def assign_user_ids(config):
    total = config["total_users"]
    min_shared = config["min_shared_users"]

    # Create at least `min_shared` users in A+B+C
    user_ids = {"ABC": [str(uuid.uuid4()) for _ in range(min_shared)]}

    remaining = total - min_shared
    combinations = ["A", "B", "C", "AB", "AC", "BC"]
    per_group = remaining // len(combinations)

    for combo in combinations:
        user_ids[combo] = [str(uuid.uuid4()) for _ in range(per_group)]

    # Fill any remaining users (due to rounding)
    leftover = remaining - (per_group * len(combinations))
    for _ in range(leftover):
        pick = random.choice(combinations)
        user_ids[pick].append(str(uuid.uuid4()))

    return user_ids

# Generate data rows based on org schema
def generate_data(user_ids):
    org_data = {"A": [], "B": [], "C": []}

    def random_age(): return random.randint(30, 80)
    def random_income(): return random.randint(100_000, 25_000_000)
    def random_history(): return random.randint(0, 50)
    def random_visits(): return random.randint(0, 100)
    def random_binary_label(income):
        # Simple heuristic: higher income → higher chance of will_purchase = 1
        probability = min(0.9, max(0.1, income / 25_000_000))
        return 1 if random.random() < probability else 0

    for combo, ids in user_ids.items():
        for uid in ids:
            if 'A' in combo:
                org_data['A'].append([
                    uid,
                    random_age(),
                    income := random_income(),
                    random_binary_label(income)
                ])
            if 'B' in combo:
                org_data['B'].append([
                    uid,
                    random_history()
                ])
            if 'C' in combo:
                org_data['C'].append([
                    uid,
                    random_visits()
                ])

    return org_data

# Write CSVs to disk
def write_csvs(data, folder, config):
    os.makedirs(folder, exist_ok=True)
    for org, rows in data.items():
        filename = os.path.join(folder, f"Org{org}.csv")
        with open(filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(config["orgs"][org])
            writer.writerows(rows)
        print(f"✅ Org{org}.csv generated with {len(rows)} rows.")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for Org A, B, and C")
    parser.add_argument("--folder", type=str, required=True, help="Name of the output folder")
    args = parser.parse_args()
    
    user_ids = assign_user_ids(CONFIG)
    data = generate_data(user_ids)
    write_csvs(data, args.folder, CONFIG)

if __name__ == "__main__":
    main()
