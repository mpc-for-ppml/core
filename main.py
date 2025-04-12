import sys
import time
from mpyc.runtime import mpc
from modules.mpc.linear import secure_linear_regression
from modules.psi.multiparty_psi import run_n_party_psi
from modules.psi.party import Party
from utils.data_loader import load_party_data_adapted
from utils.visualization import plot_actual_vs_predicted

def print_usage():
    print("Usage: python main.py [MPyC options] <dataset.csv>")
    print("\nArguments:")
    print("  [MPyC options]   : Optional, like -M (number of parties) or -I (party id)")
    print("  <dataset.csv>    : Path to the local party's CSV file")
    print("\nExample:")
    print("  python main.py -M3 -I0 party0_data.csv")
    print("  python main.py -M3 -I1 party1_data.csv")
    print("  python main.py -M3 -I2 party2_data.csv\n")
    sys.exit(1)

async def main():
    if len(sys.argv) < 2 or sys.argv[-1].startswith("-"):
        print_usage()

    # Load local party data
    csv_file = sys.argv[-1]
    party_id = mpc.pid
    user_ids, X_local, y_local, feature_names, label_name = load_party_data_adapted(csv_file)

    # Start MPC runtime
    await mpc.start()

    if y_local is None and party_id == 0:
        print(f"[Party {party_id}] ❗ Warning: Expected label missing for Org A")
    elif y_local is not None and party_id != 0:
        print(f"[Party {party_id}] ❗ Warning: Label provided but will be ignored")
        
    # Send your local feature names to all other parties
    feature_names_all = await mpc.transfer(feature_names, senders=range(len(mpc.parties)))
    
    # Broadcast label name (only Party 0 has it)
    label_name_all = await mpc.transfer(label_name, senders=[0])
    label_name = label_name_all[0] if label_name_all else "Label"

    # Flatten in party order: assume feature_names_all[i] is from party i
    joined_feature_names = []
    for f_list in feature_names_all:
        joined_feature_names.extend(f_list)

    # Step 1: Private Set Intersection (PSI) - Find common user IDs across all parties
    # Step 1.1: Collect user ID lists from all parties
    gathered_user_ids = await mpc.transfer(user_ids, senders=range(len(mpc.parties)))
    print(f"[Party {mpc.pid}] ✅ Received user ID lists from all parties.")

    # Step 1.2: Create Party instances for each list of user IDs
    parties = [Party(party_id, ids) for party_id, ids in enumerate(gathered_user_ids)]

    # Step 1.3: Run PSI to find the shared user IDs
    print(f"[Party {party_id}] 🔎 Computing intersection of user IDs...")
    start_time = time.time()
    intersection = run_n_party_psi(parties)
    elapsed_time = time.time() - start_time
    print(f"[Party {party_id}] 🔗 Found intersected user IDs in {elapsed_time:.2f}s: {intersection}")
    
    # Step 2: Join attributes for intersecting users only
    print(f"\n[Party {party_id}] 🧩 Filtering data for intersected user IDs...")

    # Step 2.1: Create a mapping from user_id to index for filtering
    id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
    intersecting_indices = [id_to_index[uid] for uid in intersection if uid in id_to_index]

    # Step 2.2: Filter local features and labels (if any)
    X_filtered = [X_local[i] for i in intersecting_indices]
    y_filtered = [y_local[i] for i in intersecting_indices] if y_local is not None else None

    print(f"[Party {party_id}] 📦 Filtered {len(X_filtered)} records.")

    # Step 2.3: Transfer X and y across all parties
    X_joined = await mpc.transfer(X_filtered, senders=range(len(mpc.parties)))
    y_final = await mpc.transfer(y_filtered, senders=[0])

    # Step 2.4: Flatten and consolidate feature vectors
    X_all = []
    y_all = []

    for i in range(len(intersection)):
        features = []
        for party_features in X_joined:
            features.extend(party_features[i])
        X_all.append(features)
        y_all.append(y_final[0][i])

    print(f"[Party {party_id}] ✅ Completed data join.")
    
    # [Bonus] Step 2.5: Pretty print the final joined data    
    print(f"\n[Party {party_id}] 🧾 Final joined dataset (features + label):")

    # Combine features and label to determine column widths
    all_rows = []
    for features, label in zip(X_all, y_all):
        row = list(map(str, features)) + [str(round(label, 2))]
        all_rows.append(row)

    # Get all column names (features + Label)
    label_name = label_name or "Label"  # fallback if somehow None
    all_headers = joined_feature_names + [label_name]

    # Calculate max width for each column
    col_widths = []
    for col_idx in range(len(all_headers)):
        max_data_len = max(len(row[col_idx]) for row in all_rows)
        header_len = len(all_headers[col_idx])
        col_widths.append(max(max_data_len, header_len) + 2)

    # Create header line
    header = "idx".ljust(5) + "| " + " | ".join(
        [all_headers[i].ljust(col_widths[i]) for i in range(len(all_headers))]
    )
    separator = "-" * len(header)

    # Print header
    print(header)
    print(separator)

    # Print data rows
    for idx, row in enumerate(all_rows):
        row_str = " | ".join(
            [row[i].ljust(col_widths[i]) for i in range(len(row))]
        )
        print(str(idx).ljust(5) + "| " + row_str)

    # At this point:
    # X_all = [ [age, income, purchase_history, web_visits], ... ] for intersecting users
    # y_all = [ purchase_amount, ... ] only from Org A

    # Step 3: Do linear regression
    # Step 3.1: Add bias coeff to X
    X_all = [row + [1.0] for row in X_all]
    
    # Step 3.2: Run the regression
    print(f"\n[Party {party_id}] ⚙️ Running linear regression to the data...")
    theta = await secure_linear_regression([X_all], [y_all])  # match expected arg shape

    # Step 4: Output result
    print(f"[Party {party_id}] ✅ Final theta (model weights): {theta}")

    # Step 5: Only visualize if you are party 0    
    if mpc.pid == 0:
        print(f"\n[Party {party_id}] 📊 Visualizing results (Only on Party 0)...")
        plot_actual_vs_predicted(X_all, y_all, theta)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
