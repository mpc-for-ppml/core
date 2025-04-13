import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import argparse

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["y"])  # Features
    y = df["y"]                 # Label (0 or 1)
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Train logistic regression on given CSV data.")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    args = parser.parse_args()

    # Load and split the dataset
    X, y = load_data(args.file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    print("\n🧠 Model Weights (theta):")
    print("Coefficients:", model.coef_[0])
    print("Intercept:", model.intercept_[0])

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {acc * 100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
