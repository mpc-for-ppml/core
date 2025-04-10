import sys
from mpyc.runtime import mpc
from data_loader import load_party_data
from regression import secure_linear_regression

async def main():
    if len(sys.argv) < 2:
        print("Usage: python secure_linreg.py -Mx -Iy dataset.csv")
        sys.exit(1)

    # Load local party data
    csv_file = sys.argv[1]
    X_local, y_local = load_party_data(csv_file)

    # Start MPC runtime
    await mpc.start()

    # Broadcast data to all parties securely
    X_all = await mpc.gather(mpc.transfer(X_local))
    y_all = await mpc.gather(mpc.transfer(y_local))

    # Run secure regression
    theta = await secure_linear_regression(X_all, y_all)

    # Output result
    print(f"[Party {mpc.pid}] Final theta (model weights): {theta}")
    await mpc.shutdown()

if __name__ == '__main__':
    mpc.run(main())
