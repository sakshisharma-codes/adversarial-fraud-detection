from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.train import train_model
from src.attack import generate_adversarial_sample
from src.detect import (
    generate_attacked_dataset,
    train_attack_detector,
    evaluate_attack_detector
)

from sklearn.model_selection import train_test_split

# Load data
df = load_data("data/creditcard.csv")

# Preprocess
df = preprocess_data(df)

# Split features and target
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# Split first (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train fraud model
model, feature_importance = train_model(X_train, y_train)

# Attack analysis (on test data)
generate_adversarial_sample(model, X_test, y_test, feature_importance)

# Generate attacked training data
attacked_train = generate_attacked_dataset(model, X_train, y_train, feature_importance)

# Train detector
detector_model = train_attack_detector(X_train, attacked_train)

print("\nEvaluating detector on unseen attacks...")

# Generate attacked test data
attacked_test = generate_attacked_dataset(model, X_test, y_test, feature_importance)

# Evaluate properly (NO retraining)
evaluate_attack_detector(detector_model, X_test, attacked_test)