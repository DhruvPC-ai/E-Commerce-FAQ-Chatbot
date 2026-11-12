"""
TensorFlow-free inference for the chatbot.

The web app only needs to *run* the trained model, not train it, and running
it is just matrix arithmetic. This module reproduces Chatbot.keras's forward
pass with NumPy, using weights exported to chatbot_model.npz by Chatbot.py.
That keeps the deployed app at ~60 MB of RAM instead of ~600 MB with TensorFlow,
so it fits on small hosting plans such as Render's free tier.

Only numpy and the standard library are used here.
"""

import json
import os
import random
import re

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUMPY_MODEL_FILE = os.path.join(BASE_DIR, "chatbot_model.npz")
DATA_FILE = os.path.join(BASE_DIR, "data.json")

CONFIDENCE_THRESHOLD = 0.45
FALLBACK_RESPONSE = ("I'm not sure I understood that. I can help with orders, shipping, "
                     "returns, payments, discounts, gifts and product availability. "
                     "Could you rephrase your question?")

# Same punctuation Keras's TextVectorization removes ("lower_and_strip_punctuation")
_STRIP_PUNCTUATION = re.compile(r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']')


def clean_text(text):
    """Light normalisation applied identically at training and inference time."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_responses(path=DATA_FILE):
    """intent name -> list of responses (also accepts the old single "response" key)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    responses = {}
    for intent in data["intents"]:
        r = intent.get("responses", intent.get("response", []))
        responses.setdefault(intent["intent"], []).extend([r] if isinstance(r, str) else r)
    return responses


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class IntentModel:
    """NumPy copy of the Keras network built in Chatbot.build_model()."""

    def __init__(self, path=NUMPY_MODEL_FILE):
        with np.load(path, allow_pickle=False) as z:
            self.w = {k: z[k] for k in z.files}
        self.classes = [str(c) for c in self.w.pop("classes")]
        self.max_len = int(self.w.pop("max_len"))
        word_vocab = [str(v) for v in self.w.pop("word_vocab")]
        char_vocab = [str(v) for v in self.w.pop("char_vocab")]
        self.word_index = {tok: i for i, tok in enumerate(word_vocab)}   # 0 = padding, 1 = [UNK]
        self.char_index = {tok: i for i, tok in enumerate(char_vocab)}   # 0 = [UNK]
        self.char_vocab_size = len(char_vocab)
        self.units = self.w["lstm_fw_recurrent"].shape[0]

    # --- preprocessing (mirrors TextVectorization) -------------------------
    @staticmethod
    def _standardize(text):
        return _STRIP_PUNCTUATION.sub("", text.lower())

    def _word_ids(self, text):
        tokens = self._standardize(text).split()[: self.max_len]
        return [self.word_index.get(t, 1) for t in tokens]

    def _char_multi_hot(self, text):
        chars = list(self._standardize(text))
        vec = np.zeros(self.char_vocab_size, dtype=np.float32)
        for i in range(len(chars) - 2):
            vec[self.char_index.get(" ".join(chars[i:i + 3]), 0)] = 1.0
        return vec

    # --- layers -------------------------------------------------------------
    def _lstm(self, x, prefix):
        """Run one LSTM direction over x (steps, features); returns (steps, units)."""
        k, rk, b = (self.w[f"{prefix}_kernel"], self.w[f"{prefix}_recurrent"],
                    self.w[f"{prefix}_bias"])
        u = self.units
        h = np.zeros(u, dtype=np.float32)
        c = np.zeros(u, dtype=np.float32)
        out = np.zeros((len(x), u), dtype=np.float32)
        xw = x @ k + b
        for t in range(len(x)):
            z = xw[t] + h @ rk
            i, f, g, o = _sigmoid(z[:u]), _sigmoid(z[u:2 * u]), np.tanh(z[2 * u:3 * u]), _sigmoid(z[3 * u:])
            c = f * c + i * g
            h = o * np.tanh(c)
            out[t] = h
        return out

    def predict_proba(self, text):
        """Probability for every intent, in the same order as self.classes."""
        w = self.w
        ids = self._word_ids(text)
        n = len(ids)

        # Word branch: embedding -> BiLSTM -> [max pool, average pool]
        if n:
            emb = w["embedding"][ids]
            fw = self._lstm(emb, "lstm_fw")
            bw = self._lstm(emb[::-1], "lstm_bw")[::-1]
            seq = np.concatenate([fw, bw], axis=1)
        else:
            seq = np.zeros((0, 2 * self.units), dtype=np.float32)
        # Keras outputs zeros at padded positions, and max pooling sees them too
        max_pool = seq.max(axis=0) if n else np.zeros(2 * self.units, dtype=np.float32)
        if n < self.max_len:
            max_pool = np.maximum(max_pool, 0.0)
        avg_pool = seq.mean(axis=0) if n else np.zeros(2 * self.units, dtype=np.float32)
        word_out = np.concatenate([max_pool, avg_pool])

        # Character-trigram branch: multi-hot -> dense(relu)
        char_out = np.maximum(self._char_multi_hot(text) @ w["char_dense_kernel"]
                              + w["char_dense_bias"], 0.0)

        # Head: concatenate -> dense(relu) -> softmax
        h = np.maximum(np.concatenate([word_out, char_out]) @ w["dense_kernel"] + w["dense_bias"], 0.0)
        logits = h @ w["out_kernel"] + w["out_bias"]
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict_intent(self, user_text):
        text = clean_text(user_text)
        if not self._word_ids(text):          # only punctuation/emoji: nothing to classify
            return None, 0.0
        probs = self.predict_proba(text)
        idx = int(np.argmax(probs))
        return self.classes[idx], float(probs[idx])


def process_input(user_text, model, responses, threshold=CONFIDENCE_THRESHOLD):
    if not user_text.strip():
        return "Please type a question and I'll do my best to help."
    intent, confidence = model.predict_intent(user_text)
    if confidence < threshold or intent not in responses:
        return FALLBACK_RESPONSE
    return random.choice(responses[intent])
