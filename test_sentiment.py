from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

textes = [
    "Ce produit est excellent, je suis très satisfait.",
    "Je suis déçu, le service est mauvais.",
    "C'est correct mais sans plus."
]

for texte in textes:
    resultat = classifier(texte)
    print("Texte :", texte)
    print("Résultat :", resultat)
    print()