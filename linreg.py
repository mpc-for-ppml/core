import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import argparse

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["y"])  # Features
    y = df["y"]                 # Label (regression target)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train linear regression on given CSV data.")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    args = parser.parse_args()

    # Load and split the dataset
    X, y = load_data(args.file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\n📈 Model Weights (theta):")
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n✅ Mean Squared Error (MSE): {mse:.4f}")
    print(f"📊 R² Score: {r2:.4f}")

if __name__ == "__main__":
    main()
