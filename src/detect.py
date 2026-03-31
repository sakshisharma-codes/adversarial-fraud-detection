import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample


# -------- Generate Attacked Dataset --------
def generate_attacked_dataset(model, X, y, feature_importance):
    attacked_samples = []

    manipulable_features = [
        f for f in feature_importance.index
        if ("score" in f or "amount" in f or "velocity" in f)
    ]

    top_features = manipulable_features[:3]

    for i in range(len(X)):
        sample = X.iloc[i].copy()

        sample_df = pd.DataFrame([sample])
        pred = model.predict(sample_df)[0]

        if y.iloc[i] == 1 and pred == 1:
            attacked = sample.copy()

            for feature in top_features:
                if feature in attacked:

                    if "score" in feature:
                        attacked[feature] += 10

                    elif "velocity" in feature:
                        attacked[feature] = max(0, attacked[feature] - 1)

                    elif "amount" in feature:
                        attacked[feature] *= 0.85

            attacked_samples.append(attacked)

    attacked_df = pd.DataFrame(attacked_samples)

    if not attacked_df.empty:
        attacked_df = attacked_df[X.columns]

    return attacked_df


# -------- Train Detector --------
def train_attack_detector(X, attacked_df):
    X_normal = X.copy()
    X_normal["is_attacked"] = 0

    attacked_df = attacked_df.copy()
    attacked_df["is_attacked"] = 1

    combined = pd.concat([X_normal, attacked_df], ignore_index=True)

    normal = combined[combined["is_attacked"] == 0]
    attacked = combined[combined["is_attacked"] == 1]

    attacked_upsampled = resample(
        attacked,
        replace=True,
        n_samples=len(normal),
        random_state=42
    )

    balanced = pd.concat([normal, attacked_upsampled])

    X_train = balanced.drop("is_attacked", axis=1)
    y_train = balanced["is_attacked"]

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# -------- Evaluate Detector --------
def evaluate_attack_detector(model, X, attacked_df):
    X_normal = X.copy()
    X_normal["is_attacked"] = 0

    attacked_df = attacked_df.copy()
    attacked_df["is_attacked"] = 1

    combined = pd.concat([X_normal, attacked_df], ignore_index=True)

    X_test = combined.drop("is_attacked", axis=1)
    y_test = combined["is_attacked"]

    y_pred = model.predict(X_test)

    print("\nAttack Detection Model Performance:\n")
    print(classification_report(y_test, y_pred))