import pandas as pd
import numpy as np

def generate_adversarial_sample(model, X, y, feature_importance):
    success_count = 0
    total_checked = 0

    manipulable_features = [
        f for f in feature_importance.index
        if ("score" in f or "amount" in f or "velocity" in f)
    ]

    top_features = manipulable_features[:3]

    print("\nAttacking (realistic features):", top_features)

    for i in range(len(X)):
        sample = X.iloc[i].copy()

        sample_df = pd.DataFrame([sample])
        original_pred = model.predict(sample_df)[0]

        if y.iloc[i] == 1 and original_pred == 1:
            total_checked += 1

            attacked = sample.copy()

            # RANDOMIZED ATTACK
            for feature in top_features:
                if feature in attacked:

                    if "score" in feature:
                        attacked[feature] += np.random.randint(-5, 20)

                    elif "velocity" in feature:
                        attacked[feature] = max(0, attacked[feature] - np.random.randint(0, 4))

                    elif "amount" in feature:
                        attacked[feature] *= np.random.uniform(0.6, 1.1)

            attacked_df = pd.DataFrame([attacked])[X.columns]
            new_pred = model.predict(attacked_df)[0]

            if new_pred == 0:
                success_count += 1

                if success_count == 1:
                    print("\nExample Successful Attack:")
                    print("Original:\n", sample)
                    print("\nAttacked:\n", attacked)

    print(f"\nTotal fraud samples tested: {total_checked}")
    print(f"Successful attacks: {success_count}")

    if total_checked > 0:
        print(f"Attack Success Rate: {success_count / total_checked:.2f}")