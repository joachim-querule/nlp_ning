from datasets import load_dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Chargement du dataset...")

dataset = load_dataset("SetFit/sst2")

train_dataset = dataset["train"].shuffle(seed=42).select(range(16))
test_dataset = dataset["test"].shuffle(seed=42).select(range(200))

print("Train :", len(train_dataset))
print("Test :", len(test_dataset))

print("Chargement du modèle...")

model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
    labels=["negative", "positive"],
    device="cpu"
)

args = TrainingArguments(
    batch_size=8,
    num_epochs=1,
    seed=42
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset
)

print("Début entraînement...")
trainer.train()

print("Prédiction...")
y_true = test_dataset["label"]

y_pred = model.predict(
    test_dataset["text"],
    batch_size=8,
    use_labels=False
)

y_pred = [int(pred) for pred in y_pred]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy :", accuracy)
print("Precision :", precision)
print("Recall :", recall)
print("F1 :", f1)

print("Test terminé.")
