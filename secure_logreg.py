import sys
from mpyc.runtime import mpc
from modules.mpc.logistic import secure_logistic_regression
from utils.data_loader import load_party_data

def print_usage():
    print("Usage: python secure_logreg.py [MPyC options] <dataset.csv>")
    print("\nArguments:")
    print("  [MPyC options]   : Optional, like -M (number of parties) or -I (party id)")
    print("  <dataset.csv>    : Path to the local party's CSV file")
    print("\nExample:")
    print("  python secure_logreg.py -M3 -I0 party0_data.csv")
    print("  python secure_logreg.py -M3 -I1 party1_data.csv")
    print("  python secure_logreg.py -M3 -I2 party2_data.csv\n")
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
    print(f"[Party {mpc.pid}] ⚙️ Running logistic regression to the data...")
    theta = await secure_logistic_regression([X_all], [y_all])

    # Output result
    print(f"\n[Party {mpc.pid}] ✅ Final theta (model weights): {theta}")

    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
