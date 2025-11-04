import streamlit as st
from pathlib import Path
import torch
import torch.nn.functional as F
import pickle
import json
import random
import numpy as np
from model import MLPTextGen 

# ---------------------------
# Config / Paths
# ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "mlp_leo_v1.pt"       # change to your .pt filename
META_PATH = MODELS_DIR / "mlp_leo_v1.json"     # change to your .json filename
VOCAB_PATH = Path("vocab.pkl")

# ---------------------------
# Load vocabulary
# ---------------------------
with open(VOCAB_PATH, "rb") as f:
    vocab_data = pickle.load(f)

vocab = vocab_data["vocab"]
word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]
most_common_word = vocab_data.get("most_common_word", list(vocab)[0])
vocab_size = len(vocab)

# ---------------------------
# Load model metadata
# ---------------------------
with open(META_PATH, "r") as f:
    metadata = json.load(f)

SEQ_LEN = metadata.get("seq_len", 5)
EMBED_DIM = metadata.get("embed_dim", 64)
HIDDEN_DIM = metadata.get("hidden_dim", 1024)
N_LAYERS = metadata.get("num_layers", 2)
ACTIVATION = metadata.get("activation", "relu")
DROPOUT = metadata.get("dropout", 0.5)

# ---------------------------
# Load model
# ---------------------------
model = MLPTextGen(
    seq_len=SEQ_LEN,        # <-- add this
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=N_LAYERS,
    activation=ACTIVATION,
    dropout=DROPOUT
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Next-k Words — Streamlit Text Predictor")
st.write("Using a single trained MLP model for text generation.")

# Sidebar controls
st.sidebar.header("Generation controls")

temperature = st.sidebar.slider(
    "Temperature (0 → deterministic)",
    min_value=0.0, max_value=2.0, value=0.8, step=0.01
)

k_words = st.sidebar.number_input(
    "Predict next k words",
    min_value=1, max_value=200, value=20
)

random_seed = st.sidebar.number_input(
    "Random seed (0 → random)",
    min_value=0, max_value=2**31-1, value=0
)

oov_strategy = st.sidebar.selectbox(
    "OOV handling",
    ["use index 0 (fallback)",
     "replace with most common word",
     "skip unknown words"]
)

st.sidebar.markdown("---")
st.sidebar.write(f"Model file: {MODEL_PATH.name}")

# ---------------------------
# Seed text input
# ---------------------------
seed_text = st.text_area("Enter seed text", "well prince so genoa and")

# ---------------------------
# Text generation function
# ---------------------------
def generate_text(seed_seq, num_words=k_words, temperature=1.0):
    model.eval()
    seq = seed_seq.split()

    # Set random seed
    if random_seed != 0:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    for _ in range(num_words):
        inp = []
        for w in seq[-SEQ_LEN:]:
            if w in word2idx:
                inp.append(word2idx[w])
            else:
                if oov_strategy == "use index 0 (fallback)":
                    inp.append(0)
                elif oov_strategy == "replace with most common word":
                    inp.append(word2idx.get(most_common_word, 0))
                elif oov_strategy == "skip unknown words":
                    continue

        # Pad if less than SEQ_LEN
        if len(inp) < SEQ_LEN:
            inp = [0] * (SEQ_LEN - len(inp)) + inp

        inp_tensor = torch.LongTensor([inp]).to(DEVICE)
        logits = model(inp_tensor)
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.multinomial(probs, num_samples=1).item()
        seq.append(idx2word[pred_idx])

    return " ".join(seq)

# ---------------------------
# Generate text button
# ---------------------------
if st.button("Generate Text"):
    if seed_text.strip() == "":
        st.warning("Please enter some seed text.")
    else:
        output_text = generate_text(seed_text, num_words=k_words, temperature=temperature)
        st.subheader("Generated Text")
        st.write(output_text)