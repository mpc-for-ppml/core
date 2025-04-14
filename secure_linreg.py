# secure_linreg.py

import sys
from mpyc.runtime import mpc
from modules.mpc.linear import SecureLinearRegression
from utils.cli_parser import parse_cli_args
from utils.constant import DEFAULT_EPOCHS, DEFAULT_LR
from utils.data_loader import load_party_data
from utils.data_normalizer import normalize_features
from utils.visualization import plot_actual_vs_predicted

async def main():
    args = parse_cli_args(type="secure_linreg")
    csv_file = args["csv_file"]
    normalizer_type = args["normalizer_type"]
    
    X_local, y_local = load_party_data(csv_file)
    
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

    # Broadcast data to all parties securely
    X_all_nested = await mpc.gather(mpc.transfer(X_local))
    y_all_nested = await mpc.gather(mpc.transfer(y_local))

    # Flatten
    X_all = sum(X_all_nested, [])
    y_all = sum(y_all_nested, [])
    
    # Get the learning variables (epochs and lr)
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

    # Run secure regression
    print(f"\n[Party {mpc.pid}] ⚙️ Running linear regression to the data...")
    model = SecureLinearRegression(epochs=epochs, lr=lr)
    await model.fit([X_all], [y_all])
    
    # Try to predict the train data
    predictions = await model.predict([X_all][0])

    # Only visualize if you are party 0
    await plot_actual_vs_predicted(y_all, predictions, mpc)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
