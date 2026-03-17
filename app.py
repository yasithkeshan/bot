import pickle
import numpy as np
import os
import logging
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from sklearn.metrics.pairwise import cosine_similarity

# Load saved chatbot index
with open("chatbot_index.pkl", "rb") as f:
    data = pickle.load(f)

vectorizer    = data["vectorizer"]
input_matrix  = data["input_matrix"]
inputs_flat   = data["inputs_flat"]
replies_flat  = data["replies_flat"]
pairs         = data["pairs"]

# Configurable constants
TOP_K         = 5
MIN_SCORE     = 0.2
FALLBACK_REPLY = "Mmmm"

# Clean text function (same as before)
import re
def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Reply function
def get_reply(user_input: str, top_k: int = TOP_K, return_list: bool = False):
    cleaned = clean_text(user_input)
    if not cleaned:
        return FALLBACK_REPLY

    vec    = vectorizer.transform([cleaned])
    scores = cosine_similarity(vec, input_matrix).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]
    best_score  = scores[top_indices[0]]

    if best_score < MIN_SCORE:
        return FALLBACK_REPLY

    top_scores = scores[top_indices]
    top_scores = top_scores / top_scores.sum()
    chosen_idx = np.random.choice(top_indices, p=top_scores)

    if return_list:
        return pairs[chosen_idx][1]
    else:
        return replies_flat[chosen_idx]

# Flask app
app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    incoming_msg = request.form.get("Body", "").strip()
    logging.info(f"Incoming message: {incoming_msg}")

    reply_text = get_reply(incoming_msg)
    logging.info(f"Replying with: {reply_text}")

    resp = MessagingResponse(reply_text)
    return str(resp)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
