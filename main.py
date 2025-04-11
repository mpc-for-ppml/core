import sys
from mpyc.runtime import mpc
from modules.mpc.linear import secure_linear_regression
from utils.data_loader import load_party_data
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
    X_local, y_local = load_party_data(csv_file)

    # Start MPC runtime
    await mpc.start()

    # Broadcast data to all parties securely
    X_all_nested = await mpc.gather(mpc.transfer(X_local))
    y_all_nested = await mpc.gather(mpc.transfer(y_local))

    # Flatten
    X_all = sum(X_all_nested, [])
    y_all = sum(y_all_nested, [])

    # Run secure regression
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
