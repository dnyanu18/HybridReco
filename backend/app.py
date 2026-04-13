from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import sys
import os

# =========================
# 🔥 Fix import path
# =========================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Model import Model
from DataHandler import DataHandler
from Params import args

# =========================
# FastAPI App
# =========================
app = FastAPI()

# =========================
# 🔥 CORS FIX
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load Data
# =========================
handler = DataHandler()
handler.LoadData()

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# =========================
# Load Model
# =========================
model = Model().to(device)

# 🔥 IMPORTANT: Load trained weights
model_path = os.path.join(os.path.dirname(__file__), "..", "best_model.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("🔥 Trained Model Loaded Successfully!")
else:
    print("⚠️ WARNING: best_model.pth not found. Using random model!")

model.eval()

# =========================
# Prepare Graph + Sequences
# =========================
adj = handler.torchBiAdj.to(device)
user_seq_map = handler.user_seq_map.to(device)

print("✅ Backend Ready!")

# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "API Running 🚀"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    # 🔥 Safety check
    if user_id < 0 or user_id >= args.user:
        return {"error": "Invalid user ID"}

    with torch.no_grad():
        # 🔥 Correct forward
        user_emb, item_emb = model(adj, user_seq_map)

        scores = torch.matmul(user_emb[user_id], item_emb.T)

        top_items = torch.topk(scores, 10).indices.tolist()

    return {
        "user": user_id,
        "recommendations": top_items
    }