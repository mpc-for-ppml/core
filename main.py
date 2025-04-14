import sys
import time
from mpyc.runtime import mpc
from modules.mpc.linear_gd import secure_linear_regression, DEFAULT_EPOCHS, DEFAULT_LR
from modules.mpc.logistic import secure_logistic_regression, approx_sigmoid
from modules.psi.multiparty_psi import run_n_party_psi
from modules.psi.party import Party
from utils.cli_parser import parse_cli_args
from utils.data_loader import load_party_data_adapted
from utils.data_normalizer import normalize_features
from utils.visualization import plot_actual_vs_predicted
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

async def main():    
    args = parse_cli_args(type="main")
    csv_file = args["csv_file"]
    normalizer_type = args["normalizer_type"]
    regression_type = args["regression_type"]

    party_id = mpc.pid
    user_ids, X_local, y_local, feature_names, label_name = load_party_data_adapted(csv_file)

    # Normalize features
    if normalizer_type:
        try:
            X_local = normalize_features(X_local, method=normalizer_type)
            print(f"[Normalizer] 🧪 Applied '{normalizer_type}' normalization.")
        except ValueError as e:
            print(f"[Normalizer] ❌ Normalization error: {e}")
            sys.exit(1)
    else:
        print(f"[Normalizer] ⚠️ No normalization applied.")

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
    
    # Step 3.2: Get the learning variables (epochs and lr)
    if mpc.pid == 0:
        try:
            epochs_input = input(f"\n[Party 0] ❓ Enter number of epochs (default={DEFAULT_EPOCHS}): \n >>  ").strip()
            lr_input = input(f"[Party 0] ❓ Enter learning rate (default={DEFAULT_LR}): \n >>  ").strip()

            epochs = int(epochs_input) if epochs_input else DEFAULT_EPOCHS
            lr = float(lr_input) if lr_input else DEFAULT_LR

            print(f"[Party 0] ✅ Using {epochs} epochs and {lr} learning rate.")
        except ValueError:
            print("[Party 0] ❌ Invalid input. Please enter numeric values.")
            sys.exit(1)
    else:
        epochs = None
        lr = None
        
    # Send from Party 0 to all parties
    epochs_all = await mpc.transfer(epochs, senders=[0])
    lr_all = await mpc.transfer(lr, senders=[0])

    # All parties now use the same values
    epochs = epochs_all[0]
    lr = lr_all[0]
    
    # Step 3.2: Run the regression
    print(f"\n[Party {party_id}] ⚙️ Running {regression_type} regression on the data...")    
    if regression_type == 'logistic':
        theta = await secure_logistic_regression([X_all], [y_all], epochs=epochs, lr=lr)
    else:
        theta = await secure_linear_regression([X_all], [y_all], epochs=epochs, lr=lr)

    # Step 4: Output result
    print(f"[Party {party_id}] ✅ Final theta (model weights): {theta}")

    # Step 5: Evaluation
    if regression_type == 'logistic':
        # Predict: Compute sigmoid(dot(x, theta)) for each sample
        sigmoid_outputs = []
        for x in X_all:
            # Dot product manually: sum(x_i * theta_i)
            dot = sum([a * b for a, b in zip(x, theta)])
            sigmoid = approx_sigmoid(dot)
            sigmoid_outputs.append(await mpc.output(sigmoid))

        # Binarize predictions
        binary_preds = [1 if p >= 0.5 else 0 for p in sigmoid_outputs]
        y_true = y_all
        y_pred = binary_preds

        # Generate the classification report
        report = classification_report(y_true, y_pred, zero_division=0)

        # Print the classification report
        print(f"\n[Party {mpc.pid}] 📊 Showing the evaluation report...")
        print(report)
        
        if mpc.pid == 0:            
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)

            # Plot
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Baseline (random guessing)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("AUC-ROC Curve")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
    else:
        if mpc.pid == 0:
            print(f"\n[Party {party_id}] 📊 Visualizing results (Only on Party 0)...")
            plot_actual_vs_predicted(X_all, y_all, theta)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
