import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_path = Path("results_setfit/setfit_results.csv")

df = pd.read_csv(results_path)

print(df)

plt.figure(figsize=(10, 6))

for epoch in sorted(df["epochs"].unique()):
    subset = df[df["epochs"] == epoch]
    plt.plot(
        subset["training_samples"],
        subset["f1"],
        marker="o",
        label=f"{epoch} epoch(s)"
    )

plt.title("SetFit - Evolution du F1 score")
plt.xlabel("Nombre d'échantillons d'entraînement")
plt.ylabel("F1 score")
plt.legend()
plt.grid(True)

output_path = Path("results_setfit/setfit_f1_score.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Graphique créé :", output_path)