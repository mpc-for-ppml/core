import sys
from mpyc.runtime import mpc
from data_loader import load_party_data
from regression import secure_linear_regression

def print_usage():
    print("Usage: python secure_linreg.py [MPyC options] <dataset.csv>")
    print("\nArguments:")
    print("  [MPyC options]   : Optional, like -M (number of parties) or -I (party id)")
    print("  <dataset.csv>    : Path to the local party's CSV file")
    print("\nExample:")
    print("  python secure_linreg.py -M3 -I0 party0_data.csv")
    print("  python secure_linreg.py -M3 -I1 party1_data.csv")
    print("  python secure_linreg.py -M3 -I2 party2_data.csv\n")
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
    X_all = await mpc.gather(mpc.transfer(X_local))
    y_all = await mpc.gather(mpc.transfer(y_local))

    # Run secure regression
    print("Processing the data...")
    theta = await secure_linear_regression(X_all, y_all)

    # Output result
    print(f"[Party {mpc.pid}] Final theta (model weights): {theta}")
    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
