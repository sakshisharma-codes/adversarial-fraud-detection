import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

def train_model(X, y):
    # Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Combine into one dataframe for resampling
    train_df = X_train.copy()
    train_df["is_fraud"] = y_train

    # Separate classes
    fraud = train_df[train_df["is_fraud"] == 1]
    normal = train_df[train_df["is_fraud"] == 0]

    # Upsample fraud (important)
    fraud_upsampled = resample(
        fraud,
        replace=True,
        n_samples=len(normal) // 2,  # not full balance (more realistic)
        random_state=42
    )

    # Combine back
    balanced = pd.concat([normal, fraud_upsampled])

    # Split again
    X_train_bal = balanced.drop("is_fraud", axis=1)
    y_train_bal = balanced["is_fraud"]

    # Train model
    model = RandomForestClassifier(
        class_weight="balanced",
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train_bal, y_train_bal)

    # --- IMPORTANT: threshold tuning ---
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.3).astype(int)

    print("\nModel Performance:\n")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    print("\nTop 5 Important Features:\n")
    print(feature_importance.head(5))

    return model, feature_importance