import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import load_dataset, concatenate_datasets, Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==========================================================
# CONFIGURATION GÉNÉRALE
# ==========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Important : on force le CPU.
# Ta carte GT 730 est détectée mais trop ancienne pour PyTorch moderne.
DEVICE = "cpu"

DATASET_NAME = "SetFit/sst2"

# Modèle léger pour ton PC.
# Il est plus rapide que les gros modèles comme all-mpnet-base-v2.
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2"

TRAINING_SAMPLES_LIST = [8, 10, 20, 50, 100]
EPOCHS_LIST = [1, 5, 10]

RESULTS_DIR = "results_setfit"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================================
# 1. CHARGEMENT DU DATASET
# ==========================================================

print("Chargement du dataset Hugging Face...")
dataset = load_dataset(DATASET_NAME)

# On fusionne train, validation et test pour refaire notre propre split 80 / 20.
full_dataset = concatenate_datasets([
    dataset["train"],
    dataset["validation"],
    dataset["test"]
])

df = full_dataset.to_pandas()
df = df[["text", "label"]]

print("Nombre total d'exemples :", len(df))
print("Répartition des labels :")
print(df["label"].value_counts())


# ==========================================================
# 2. SPLIT 80 % / 20 %
# ==========================================================

train_eval_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED,
    stratify=df["label"]
)

train_eval_df = train_eval_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("\nTaille train + eval :", len(train_eval_df))
print("Taille test fixe :", len(test_df))

# Le test set restera exactement le même pour toutes les expériences.
test_texts = test_df["text"].tolist()
y_true = test_df["label"].tolist()


# ==========================================================
# 3. FONCTION POUR CRÉER UN PETIT DATASET FEW-SHOT
# ==========================================================

def create_balanced_few_shot_dataset(dataframe, total_samples):
    """
    Crée un dataset few-shot équilibré.

    Exemple :
    total_samples = 8
    Dataset binaire positif / négatif
    Donc 4 exemples négatifs + 4 exemples positifs.
    """

    labels = sorted(dataframe["label"].unique())
    number_of_classes = len(labels)

    if total_samples % number_of_classes != 0:
        raise ValueError("Le nombre d'échantillons doit être divisible par le nombre de classes.")

    samples_per_class = total_samples // number_of_classes

    sampled_parts = []

    for label in labels:
        label_df = dataframe[dataframe["label"] == label]

        sampled_label_df = label_df.sample(
            n=samples_per_class,
            random_state=SEED
        )

        sampled_parts.append(sampled_label_df)

    sampled_df = pd.concat(sampled_parts)
    sampled_df = sampled_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return Dataset.from_pandas(sampled_df)


# ==========================================================
# 4. LANCEMENT DES 15 EXPÉRIENCES
# ==========================================================

all_results = []

experiment_number = 1
total_experiments = len(TRAINING_SAMPLES_LIST) * len(EPOCHS_LIST)

for training_samples in TRAINING_SAMPLES_LIST:
    for epochs in EPOCHS_LIST:

        print("\n" + "=" * 70)
        print(f"EXPÉRIENCE {experiment_number}/{total_experiments}")
        print(f"Échantillons d'entraînement : {training_samples}")
        print(f"Epochs : {epochs}")
        print("=" * 70)

        train_dataset = create_balanced_few_shot_dataset(
            train_eval_df,
            total_samples=training_samples
        )

        print("Taille du train few-shot :", len(train_dataset))
        print("Labels dans le train few-shot :", train_dataset["label"])

        print("\nChargement du modèle SetFit...")

        model = SetFitModel.from_pretrained(
            MODEL_NAME,
            labels=["negative", "positive"],
            device=DEVICE
        )

        args = TrainingArguments(
            batch_size=8,
            num_epochs=epochs,
            seed=SEED
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset
        )

        print("Début de l'entraînement...")
        trainer.train()

        print("Évaluation sur le test set fixe...")

        predictions = model.predict(
            test_texts,
            batch_size=8,
            use_labels=False
        )

        y_pred = [int(pred) for pred in predictions]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)

        result = {
            "experiment": experiment_number,
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,
            "training_samples": training_samples,
            "epochs": epochs,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4)
        }

        all_results.append(result)

        print("\nRésultat de cette expérience :")
        print(result)

        # Sauvegarde progressive pour ne pas perdre les résultats
        temp_results_df = pd.DataFrame(all_results)
        temp_results_df.to_csv(
            os.path.join(RESULTS_DIR, "setfit_results_progress.csv"),
            index=False
        )

        experiment_number += 1


# ==========================================================
# 5. SAUVEGARDE FINALE DES RÉSULTATS
# ==========================================================

results_df = pd.DataFrame(all_results)

csv_path = os.path.join(RESULTS_DIR, "setfit_results.csv")
excel_path = os.path.join(RESULTS_DIR, "setfit_results.xlsx")

results_df.to_csv(csv_path, index=False)
results_df.to_excel(excel_path, index=False)

print("\n" + "=" * 70)
print("TOUTES LES EXPÉRIENCES SONT TERMINÉES")
print("=" * 70)

print("\nTableau final :")
print(results_df)

print("\nFichiers créés :")
print(csv_path)
print(excel_path)
