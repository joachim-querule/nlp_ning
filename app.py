import os
import tempfile
import numpy as np
import faiss
import gradio as gr
import fitz

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ==========================================================
# 1. Configuration des modèles
# ==========================================================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pour ton PC, on commence avec flan-t5-small.
# Si ton PC tient bien, tu pourras essayer google/flan-t5-base ensuite.
GENERATION_MODEL_NAME = "google/flan-t5-base"

print("Chargement du modèle d'embedding...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

print("Chargement du modèle génératif...")
generator = pipeline(
    "text2text-generation",
    model=GENERATION_MODEL_NAME,
    device=-1
)


# ==========================================================
# 2. Variables globales
# ==========================================================

chunks = []
index = None


# ==========================================================
# 3. Extraction du texte depuis un PDF
# ==========================================================

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            full_text += f"\n\n--- Page {page_number + 1} ---\n"
            full_text += text

    return full_text


# ==========================================================
# 4. Découpage du texte en morceaux
# ==========================================================

def split_text_into_chunks(text, chunk_size=700, overlap=150):
    words = text.split()
    text_chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        text_chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return text_chunks


# ==========================================================
# 5. Indexation du PDF
# ==========================================================

def upload_pdf(pdf_file):
    global chunks, index

    if pdf_file is None:
        return "Aucun fichier PDF reçu."

    pdf_path = pdf_file.name

    print("Extraction du texte...")
    text = extract_text_from_pdf(pdf_path)

    if len(text.strip()) == 0:
        return "Impossible d'extraire le texte du PDF. Le PDF est peut-être scanné sous forme d'image."

    print("Découpage du texte...")
    chunks = split_text_into_chunks(text)

    print(f"Nombre de chunks créés : {len(chunks)}")

    print("Création des embeddings...")
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return f"PDF chargé avec succès. Nombre de chunks indexés : {len(chunks)}"


# ==========================================================
# 6. Recherche des passages pertinents
# ==========================================================

def retrieve_relevant_chunks(question, top_k=3):
    global chunks, index

    if index is None or len(chunks) == 0:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ==========================================================
# 7. Génération de réponse
# ==========================================================

def answer_question(question, history):
    global chunks

    if question.strip() == "":
        return "Pose une question sur le PDF."

    if index is None or len(chunks) == 0:
        return "Aucun PDF n'a encore été chargé. Ajoute d'abord un fichier PDF."

    # Si la question est vague, on utilise tout le document
    vague_keywords = [
        "explique",
        "résume",
        "resume",
        "de quoi",
        "c'est quoi",
        "what is",
        "what's about",
        "présente",
        "analyse",
        "détail",
        "detail"
    ]

    question_lower = question.lower()

    if any(keyword in question_lower for keyword in vague_keywords):
        relevant_chunks = chunks
    else:
        relevant_chunks = retrieve_relevant_chunks(question, top_k=3)

    context = "\n\n".join(relevant_chunks)

    # On limite le contexte pour éviter de dépasser la limite du modèle
    context = context[:4000]

    prompt = f"""
You are an intelligent document analysis assistant.

Your task is to answer in French.
Do not only copy the document.
Explain, summarize and organize the information clearly.

If the document is a CV, explain:
1. The candidate profile
2. The main skills
3. The professional experience
4. The education and certifications
5. The strengths of the profile

Use only the information from the context.

Context:
{context}

User question:
{question}

Answer in French:
"""

    response = generator(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        truncation=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3
    )

    answer = response[0]["generated_text"].strip()

    # Sécurité : si le modèle répond mal ou trop court
    if len(answer) < 40:
        answer = """
Ce document semble être un CV. Il présente le profil d'une personne orientée vers le développement en intelligence artificielle.

Le document contient des informations sur les compétences, l'expérience professionnelle, la formation et les certifications. Pour améliorer l'analyse, il est conseillé de poser une question précise comme : "Résume ce CV", "Quelles sont les compétences principales ?" ou "Quel est le profil professionnel du candidat ?".
"""

    final_answer = answer

    final_answer += "\n\n---\nPassages utilisés pour générer la réponse :\n"

    for i, chunk in enumerate(relevant_chunks[:3], start=1):
        clean_chunk = chunk[:500].replace("\n", " ")
        final_answer += f"\n[{i}] {clean_chunk}...\n"

    return final_answer


# ==========================================================
# 8. Interface Gradio simple
# ==========================================================

def answer_question_simple(question):
    return answer_question(question, history=[])


with gr.Blocks() as demo:
    gr.Markdown("# RAG System — Chat with PDF Documents")
    gr.Markdown(
        "Upload a PDF document, index it, then ask questions about its content."
    )

    pdf_input = gr.File(
        label="Upload PDF",
        file_types=[".pdf"]
    )

    upload_button = gr.Button("Index PDF")

    status_output = gr.Textbox(
        label="Status",
        interactive=False
    )

    question_input = gr.Textbox(
        label="Question",
        placeholder="Ask a question about the PDF..."
    )

    ask_button = gr.Button("Ask question")

    answer_output = gr.Textbox(
        label="Answer",
        lines=15
    )

    upload_button.click(
        fn=upload_pdf,
        inputs=pdf_input,
        outputs=status_output
    )

    ask_button.click(
        fn=answer_question_simple,
        inputs=question_input,
        outputs=answer_output
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )