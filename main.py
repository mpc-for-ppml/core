import sys
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
    user_ids, X_local, y_local = load_party_data_adapted(csv_file, party_id)

    # Start MPC runtime
    await mpc.start()

    if y_local is None and party_id == 0:
        print(f"[Party {party_id}] ❗ Warning: Expected label missing for Org A")
    elif y_local is not None and party_id != 0:
        print(f"[Party {party_id}] ❗ Warning: Label provided but will be ignored")

    # TODO:
    # Step 1: PSI - Determine shared user_ids across all parties
    # Step 2: Join attributes for intersecting users only
    filtered_X_local, filtered_y_local = [], []

    # Step 3: Gather all X and y from parties
    X_all_nested = await mpc.gather(mpc.transfer(filtered_X_local))
    y_all_nested = await mpc.gather(mpc.transfer(filtered_y_local if y_local else []))

    # Flatten
    X_all = sum(X_all_nested, [])
    y_all = sum(y_all_nested, [])

    # At this point:
    # X_all = [ [age, income, purchase_history, web_visits], ... ] for intersecting users
    # y_all = [ purchase_amount, ... ] only from Org A

    if not X_all or not y_all:
        print(f"[Party {party_id}] ⚠️ No common data points after PSI. Exiting.")
        await mpc.shutdown()
        return

    # Step 4: Run regression
    print("Processing the data...")
    theta = await secure_linear_regression([X_all], [y_all])  # match expected arg shape

    # Output result
    print(f"[Party {mpc.pid}] Final theta (model weights): {theta}")

    # Only visualize if you are party 0
    if mpc.pid == 0:
        print("Visualizing results (only on Party 0)...")
        plot_actual_vs_predicted(X_all, y_all, theta)

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
