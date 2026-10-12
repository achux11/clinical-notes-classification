"""
Clinical Notes AI - covers: clinical NLP, deep learning, PyTorch, fine-tuning, LangGraph, subagents
Dataset: MTSamples (~5K de-identified medical transcriptions) - Kaggle "Medical Transcriptions"
         (tboyle10/medicaltranscriptions). Download mtsamples.csv and upload it.

Run in Google Colab with a GPU (Runtime -> Change runtime type -> T4 GPU).

  Cell 1:  !pip install -q transformers datasets scikit-learn langgraph langchain-community \
                          langchain-huggingface chromadb sentence-transformers langchain-groq
  Cell 2:  from google.colab import files; files.upload()      # pick mtsamples.csv
  Cell 3:  paste this whole file and run it.

PART A (PyTorch fine-tuning) needs only the GPU.
PART B (LangGraph agent) needs a free Groq API key (console.groq.com) for the LLM:
  import os; os.environ["GROQ_API_KEY"] = "your_key"
Write down every number printed under RESULTS - those go on your resume.
"""
import os, re, random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ============ 1. Load + clean ============
df = pd.read_csv("mtsamples.csv").dropna(subset=["transcription", "medical_specialty"])
df["medical_specialty"] = df["medical_specialty"].str.strip()
# Drop catch-all / administrative categories that are not real clinical specialties
drop = {"Surgery", "Consult - History and Phy.", "SOAP / Chart / Progress Notes",
        "Discharge Summary", "Office Notes", "Letters", "General Medicine", "Emergency Room Reports"}
df = df[~df["medical_specialty"].isin(drop)]
top = df["medical_specialty"].value_counts().head(6).index      # 6 largest specialties
df = df[df["medical_specialty"].isin(top)].reset_index(drop=True)
df["text"] = df["transcription"].str.replace(r"\s+", " ", regex=True).str[:3000]
labels = sorted(df["medical_specialty"].unique()); lab2id = {l: i for i, l in enumerate(labels)}
df["label"] = df["medical_specialty"].map(lab2id)
print(f"Notes: {len(df)} | Specialties ({len(labels)}): {labels}")

train, test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
train, val = train_test_split(train, test_size=0.125, stratify=train["label"], random_state=SEED)

# ============ 2. Baseline: TF-IDF + logistic regression ============
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
base = LogisticRegression(max_iter=2000, class_weight="balanced")
base.fit(tfidf.fit_transform(train["text"]), train["label"])
base_f1 = f1_score(test["label"], base.predict(tfidf.transform(test["text"])), average="macro")

# ============ PART A: fine-tune DistilBERT in PyTorch ============
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "distilbert-base-uncased"
tok = AutoTokenizer.from_pretrained(MODEL)

def make_loader(frame, shuffle):
    enc = tok(list(frame["text"]), truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(frame["label"].values))
    return DataLoader(ds, batch_size=16, shuffle=shuffle)

tr_dl, va_dl, te_dl = make_loader(train, True), make_loader(val, False), make_loader(test, False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(labels)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

def predict(dl):
    model.eval(); preds = []
    with torch.no_grad():
        for ids, mask, _ in dl:
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            preds += logits.argmax(-1).cpu().tolist()
    return np.array(preds)

best_f1, best_state, EPOCHS = -1, None, 4
for epoch in range(EPOCHS):
    model.train()
    for ids, mask, y in tr_dl:
        loss = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=y.to(device)).loss
        opt.zero_grad(); loss.backward(); opt.step()
    va_f1 = f1_score(val["label"], predict(va_dl), average="macro")
    print(f"epoch {epoch+1}: val macro-F1 {va_f1:.3f}")
    if va_f1 > best_f1:                                   # keep best epoch (chosen on VALIDATION)
        best_f1 = va_f1
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)
te_pred = predict(te_dl)                                  # locked test set, evaluated once
bert_f1 = f1_score(test["label"], te_pred, average="macro")

# ============ PART B: LangGraph agent (retriever -> answerer -> verifier subagents) ============
agent_acc, n_q = None, 0
if os.getenv("GROQ_API_KEY"):
    from typing import TypedDict, List
    from langgraph.graph import StateGraph, END
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docs = list(train["text"].str[:1500])
    metas = [{"specialty": s, "sample": n} for s, n in zip(train["medical_specialty"], train["sample_name"])]
    store = Chroma.from_texts(docs, emb, metadatas=metas)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    class S(TypedDict):
        question: str; context: List[str]; specialties: List[str]; answer: str; grounded: bool; tries: int

    def retriever(s):                                     # subagent 1: find similar notes
        hits = store.similarity_search(s["question"], k=5)
        return {"context": [h.page_content for h in hits], "specialties": [h.metadata["specialty"] for h in hits]}

    def answerer(s):                                      # subagent 2: answer only from retrieved notes
        ctx = "\n---\n".join(f"[{sp}] {c[:600]}" for sp, c in zip(s["specialties"], s["context"]))
        prompt = (f"Using ONLY these similar clinical notes:\n{ctx}\n\nQuestion: {s['question']}\n"
                  f"Answer with exactly one specialty from: {labels}.")
        return {"answer": llm.invoke(prompt).content.strip(), "tries": s.get("tries", 0) + 1}

    def verifier(s):                                      # subagent 3: reject ungrounded / invalid answers
        ok = any(l.lower() in s["answer"].lower() for l in labels) and \
             any(l.lower() in s["answer"].lower() for l in s["specialties"])
        return {"grounded": ok}

    g = StateGraph(S)
    g.add_node("retriever", retriever); g.add_node("answerer", answerer); g.add_node("verifier", verifier)
    g.set_entry_point("retriever"); g.add_edge("retriever", "answerer"); g.add_edge("answerer", "verifier")
    g.add_conditional_edges("verifier", lambda s: END if s["grounded"] or s["tries"] >= 2 else "answerer")
    app = g.compile()

    # Evaluation set: 50 held-out TEST notes; question = "which specialty?" from the note's description
    q = test.sample(50, random_state=SEED)
    correct = 0
    for _, row in q.iterrows():
        out = app.invoke({"question": f"Which specialty handles this case: {row['description']}", "tries": 0})
        pred = next((l for l in labels if l.lower() in out["answer"].lower()), None)
        correct += pred == row["medical_specialty"]
    n_q, agent_acc = len(q), correct / len(q)

# ============ RESULTS (put these on the resume) ============
print("\n================ RESULTS ================")
print(f"Notes used: {len(df)} | Specialties: {len(labels)} | Test notes: {len(test)}")
print(f"Baseline TF-IDF + LogReg  test macro-F1: {base_f1:.3f}")
print(f"DistilBERT (PyTorch)      test macro-F1: {bert_f1:.3f}")
print(classification_report(test["label"], te_pred, target_names=labels, digits=3))
if agent_acc is not None:
    print(f"LangGraph agent accuracy on {n_q}-question test set: {agent_acc:.1%}")
else:
    print("Part B skipped: set GROQ_API_KEY to run the LangGraph agent.")
