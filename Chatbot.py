"""
Intent-classification chatbot: training (TensorFlow / Keras 3) + terminal chat.

Usage
-----
  python Chatbot.py                    # chat in the terminal (trains first if no model exists)
  python Chatbot.py --train            # (re)train the model from data.json, then chat
  python Chatbot.py --train --no-chat  # retrain only
  python Chatbot.py --export           # rebuild chatbot_model.npz from Chatbot.keras
  python Chatbot.py --debug            # chat, showing predicted intent + confidence

Files
-----
  data.json          training data  {"intents": [{"intent", "examples", "responses"}]}
  Chatbot.keras      trained Keras model (text preprocessing is built in)
  classes.json       intent label order used by the model's output layer
  chatbot_model.npz  the same model's weights for inference.py (NumPy only, no
                     TensorFlow). Written automatically after training; this is
                     what the web app (app.py) uses.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import re
import warnings

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from inference import (CONFIDENCE_THRESHOLD, NUMPY_MODEL_FILE, IntentModel,
                       clean_text, load_responses, process_input)

# GlobalMaxPooling1D ignores the padding mask; that's harmless here because the
# LSTM outputs zeros on padded steps (inference.py reproduces this exactly).
warnings.filterwarnings("ignore", message=".*does not support masking.*")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_FILE = "data.json"
MODEL_FILE = "Chatbot.keras"
CLASSES_FILE = "classes.json"

SEED = 42
MAX_LEN = 32              # tokens per sentence (longest example is ~25 words)
MAX_WORD_TOKENS = 6000    # word vocabulary cap
MAX_CHAR_TOKENS = 8000    # character-trigram vocabulary cap
EMBED_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 128
DROPOUT = 0.4

BATCH_SIZE = 32           # small batches -> ~100 updates/epoch instead of 4
MAX_EPOCHS = 80           # upper bound; early stopping picks the real number
LEARNING_RATE = 1e-3
LABEL_SMOOTHING = 0.1     # keeps confidences calibrated so the threshold works
VAL_SPLIT = 0.15
AUGMENT_COPIES = 2        # noisy copies of each training sentence
# CONFIDENCE_THRESHOLD and the fallback reply live in inference.py


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_data(path=DATA_FILE):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes, texts, labels = [], [], []
    for intent in data["intents"]:
        name = intent["intent"]
        if name not in classes:
            classes.append(name)
        idx = classes.index(name)
        for example in intent["examples"]:
            texts.append(clean_text(example))
            labels.append(idx)
    return texts, np.array(labels), classes


def stratified_split(labels, val_frac, rng):
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


KEYBOARD = "abcdefghijklmnopqrstuvwxyz"


def augment_sentence(text, rng):
    """Create a noisy variant: drop a word, swap neighbours, or add a typo."""
    words = text.split()
    if len(words) < 2:
        return text
    op = rng.integers(0, 4)
    if op == 0 and len(words) > 3:                      # random word deletion
        del words[rng.integers(0, len(words))]
    elif op == 1:                                       # swap two adjacent words
        i = rng.integers(0, len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    elif op == 2:                                       # single-character typo
        i = rng.integers(0, len(words))
        w = words[i]
        if len(w) > 3:
            j = rng.integers(1, len(w) - 1)
            kind = rng.integers(0, 3)
            if kind == 0:
                w = w[:j] + w[j + 1:]                   # deletion
            elif kind == 1:
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:] # transposition
            else:
                w = w[:j] + KEYBOARD[rng.integers(0, 26)] + w[j + 1:]
            words[i] = w
    else:                                               # strip punctuation
        return re.sub(r"[^\w\s]", "", text)
    return " ".join(words)


def augment(texts, labels, copies, rng):
    out_t, out_y = list(texts), list(labels)
    for t, y in zip(texts, labels):
        variants = {augment_sentence(t, rng) for _ in range(copies)} - {t}
        out_t.extend(variants)
        out_y.extend([y] * len(variants))
    return out_t, np.array(out_y)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(train_texts, num_classes):
    """Two-branch network.

    * Word branch: embedding -> BiLSTM -> max + average pooling. Captures word
      order ("can I return..." vs "can I cancel...", negations, etc.).
    * Character-trigram branch: bag of char 3-grams -> dense. Makes the model
      robust to typos and unseen word forms ("trackin", "delivry").
    Both preprocessing layers live inside the model, so it takes raw strings.
    """
    word_vec = layers.TextVectorization(
        max_tokens=MAX_WORD_TOKENS, output_mode="int",
        output_sequence_length=MAX_LEN, name="word_vectorizer")
    char_vec = layers.TextVectorization(
        max_tokens=MAX_CHAR_TOKENS, split="character", ngrams=(3,),
        output_mode="multi_hot", name="char_vectorizer")
    ds = tf.data.Dataset.from_tensor_slices(train_texts).batch(256)
    word_vec.adapt(ds)
    char_vec.adapt(ds)

    inputs = keras.Input(shape=(1,), dtype="string", name="text")

    # Word branch
    x = word_vec(inputs)
    x = layers.Embedding(word_vec.vocabulary_size(), EMBED_DIM, mask_zero=True)(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = layers.Concatenate()([layers.GlobalMaxPooling1D()(x),
                              layers.GlobalAveragePooling1D()(x)])

    # Character-trigram branch
    # multi_hot (not tf_idf): tf_idf layers can't be reloaded in Keras 3.10.
    # Rescaling(1.0) just casts the 0/1 integer vector to float32.
    c = layers.Flatten()(char_vec(inputs))
    c = layers.Rescaling(1.0)(c)
    c = layers.Dropout(0.3)(c)
    c = layers.Dense(DENSE_UNITS, activation="relu")(c)

    h = layers.Concatenate()([x, c])
    h = layers.Dropout(DROPOUT)(h)
    h = layers.Dense(DENSE_UNITS, activation="relu")(h)
    h = layers.Dropout(DROPOUT)(h)
    outputs = layers.Dense(num_classes, activation="softmax", name="intent")(h)

    model = keras.Model(inputs, outputs, name="intent_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"])
    return model


def class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    weights = len(labels) / (num_classes * np.maximum(counts, 1))
    return {i: float(w) for i, w in enumerate(weights)}


def to_inputs(texts):
    return tf.constant(list(texts), dtype=tf.string)[:, None]


def fit(model, texts, labels, num_classes, epochs, val=None, monitor="val_loss"):
    callbacks = [keras.callbacks.ReduceLROnPlateau(
        monitor=monitor, factor=0.5, patience=3, min_lr=1e-5, verbose=0)]
    if val is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor=monitor, patience=8, restore_best_weights=True, verbose=0))
        val = (to_inputs(val[0]), keras.utils.to_categorical(val[1], num_classes))
    return model.fit(
        to_inputs(texts), keras.utils.to_categorical(labels, num_classes),
        validation_data=val, epochs=epochs, batch_size=BATCH_SIZE, shuffle=True,
        class_weight=class_weights(labels, num_classes), callbacks=callbacks, verbose=2)


def train():
    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    texts, labels, classes = load_data()
    n = len(classes)
    print(f"Loaded {len(texts)} examples across {n} intents.")

    # Phase 1: hold out a validation set to find the best number of epochs
    tr, va = stratified_split(labels, VAL_SPLIT, rng)
    tr_t, tr_y = augment([texts[i] for i in tr], labels[tr], AUGMENT_COPIES, rng)
    va_t, va_y = [texts[i] for i in va], labels[va]
    print(f"\nPhase 1: {len(tr_t)} training (incl. augmented) / {len(va_t)} validation")
    model = build_model(tr_t, n)
    model.summary()
    hist = fit(model, tr_t, tr_y, n, MAX_EPOCHS, val=(va_t, va_y))

    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    probs = model.predict(to_inputs(va_t), verbose=0)
    preds = probs.argmax(1)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Validation accuracy: {np.mean(preds == va_y):.3f}")
    print(f"Validation accuracy on confident predictions (>= {CONFIDENCE_THRESHOLD}): "
          f"{np.mean(preds[probs.max(1) >= CONFIDENCE_THRESHOLD] == va_y[probs.max(1) >= CONFIDENCE_THRESHOLD]):.3f}"
          f" (coverage {np.mean(probs.max(1) >= CONFIDENCE_THRESHOLD):.3f})")
    weak = [(classes[c], np.mean(preds[va_y == c] == c)) for c in range(n)]
    weak = sorted(weak, key=lambda t: t[1])[:5]
    print("Weakest intents on validation:", ", ".join(f"{k} ({v:.2f})" for k, v in weak))

    # Phase 2: retrain on ALL data for the chosen number of epochs
    keras.utils.set_random_seed(SEED)
    all_t, all_y = augment(texts, labels, AUGMENT_COPIES, rng)
    print(f"\nPhase 2: final training on {len(all_t)} examples for {best_epoch} epochs")
    model = build_model(all_t, n)
    fit(model, all_t, all_y, n, best_epoch, monitor="loss")

    model.save(MODEL_FILE)
    with open(CLASSES_FILE, "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=2)
    print(f"\nSaved {MODEL_FILE} and {CLASSES_FILE}")
    return model, classes


# ----------------------------------------------------------------------------
# Export for TensorFlow-free inference (used by app.py via inference.py)
# ----------------------------------------------------------------------------
def export_numpy_model(model, classes, path=NUMPY_MODEL_FILE):
    """Save the weights + vocabularies inference.py needs, then check it matches Keras."""
    word_vec = model.get_layer("word_vectorizer")
    char_vec = model.get_layer("char_vectorizer")
    word_vocab = [str(t) for t in word_vec.get_vocabulary()]
    char_vocab = [str(t) for t in char_vec.get_vocabulary()]
    embedding = next(l for l in model.layers if isinstance(l, layers.Embedding))
    bilstm = next(l for l in model.layers if isinstance(l, layers.Bidirectional))
    out = model.get_layer("intent")
    dense = [l for l in model.layers if isinstance(l, layers.Dense) and l is not out]
    char_dense = next(l for l in dense if l.get_weights()[0].shape[0] == len(char_vocab))
    head_dense = next(l for l in dense if l is not char_dense)
    fw, bw = bilstm.forward_layer.get_weights(), bilstm.backward_layer.get_weights()
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    np.savez_compressed(
        path,
        classes=np.array(classes),
        max_len=np.array(word_vec.get_config()["output_sequence_length"]),
        word_vocab=np.array(word_vocab), char_vocab=np.array(char_vocab),
        embedding=f32(embedding.get_weights()[0]),
        lstm_fw_kernel=f32(fw[0]), lstm_fw_recurrent=f32(fw[1]), lstm_fw_bias=f32(fw[2]),
        lstm_bw_kernel=f32(bw[0]), lstm_bw_recurrent=f32(bw[1]), lstm_bw_bias=f32(bw[2]),
        char_dense_kernel=f32(char_dense.get_weights()[0]),
        char_dense_bias=f32(char_dense.get_weights()[1]),
        dense_kernel=f32(head_dense.get_weights()[0]), dense_bias=f32(head_dense.get_weights()[1]),
        out_kernel=f32(out.get_weights()[0]), out_bias=f32(out.get_weights()[1]))

    # Safety check: the NumPy copy must give the same answers as Keras
    texts, _, _ = load_data()
    np_model = IntentModel(path)
    keras_probs = model.predict(to_inputs(texts), verbose=0, batch_size=256)
    numpy_probs = np.array([np_model.predict_proba(t) for t in texts])
    max_diff = float(np.abs(keras_probs - numpy_probs).max())
    agree = float(np.mean(keras_probs.argmax(1) == numpy_probs.argmax(1)))
    print(f"Saved {path} (NumPy vs Keras on {len(texts)} examples: "
          f"same intent {agree * 100:.2f}%, max probability difference {max_diff:.1e})")
    if agree < 1.0 or max_diff > 1e-3:
        raise RuntimeError("The NumPy model does not match the Keras model. "
                           "inference.py may need updating if build_model() was changed.")


def load_trained():
    try:
        model = keras.models.load_model(MODEL_FILE, compile=False)
    except Exception as e:
        saved = "unknown"
        try:
            import zipfile
            with zipfile.ZipFile(MODEL_FILE) as z:
                saved = json.loads(z.read("metadata.json")).get("keras_version", saved)
        except Exception:
            pass
        raise RuntimeError(
            f"Could not load {MODEL_FILE} (saved with Keras {saved}, you have Keras "
            f"{keras.__version__}). Retrain it with your installed version by running:\n"
            f"    python Chatbot.py --train --no-chat\n"
            f"Original error: {e}") from e
    with open(CLASSES_FILE, "r", encoding="utf-8") as f:
        classes = json.load(f)
    return model, classes


# ----------------------------------------------------------------------------
# Terminal chat (uses the same NumPy engine as the web app)
# ----------------------------------------------------------------------------
def chat(debug=False):
    model, responses = IntentModel(), load_responses()
    print("Type 'exit' to quit.\nChatbot: Hello! How can I assist you today?")
    while True:
        try:
            user_text = input("You : ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_text.strip().lower() in {"exit", "quit"}:
            print("Chatbot: Goodbye! Take care.")
            break
        if debug:
            intent, conf = model.predict_intent(user_text)
            print(f"   [intent={intent} confidence={conf:.2f}]")
        print("Chatbot:", process_input(user_text, model, responses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent chatbot")
    parser.add_argument("--train", action="store_true", help="retrain from data.json")
    parser.add_argument("--export", action="store_true",
                        help="rebuild chatbot_model.npz from Chatbot.keras")
    parser.add_argument("--no-chat", action="store_true", help="exit after training/export")
    parser.add_argument("--debug", action="store_true", help="show predicted intent")
    args = parser.parse_args()

    if args.train or not (os.path.exists(MODEL_FILE) and os.path.exists(CLASSES_FILE)):
        export_numpy_model(*train())
    elif args.export or not os.path.exists(NUMPY_MODEL_FILE):
        export_numpy_model(*load_trained())

    if not args.no_chat:
        chat(debug=args.debug)
