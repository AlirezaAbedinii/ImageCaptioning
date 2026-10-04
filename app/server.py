"""Inference HTTP server for the captioning dashboard.

Accepts an image via ``POST /`` and returns the generated caption as plain
text. Mirrors the original ``Server.py``: it loads the trained generator,
rebuilds the tokenizer from the cached training captions, encodes the uploaded
image with InceptionV3 and decodes greedily.

Run:
    python -m app.server
The trained weights are not stored in git — drop ``final_model_V4.h5`` and
``flat_train_caps.pickle`` into the ``models/`` folder first (see README).
"""
from __future__ import annotations

import os
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config                       # noqa: E402
from src.config import CONFIG                # noqa: E402
from src.inference import generate_caption   # noqa: E402

HOST_NAME = "localhost"
SERVER_PORT = 8000
TOKENIZER_FILTERS = "!'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"


def create_tokenizer(flat_train_caps, num_words: int = CONFIG.num_words) -> Tokenizer:
    tokenizer = Tokenizer(oov_token="OOV", filters=TOKENIZER_FILTERS,
                          num_words=num_words)
    tokenizer.fit_on_texts(flat_train_caps)
    return tokenizer


# --- load artefacts once at start-up --------------------------------------- #
print("Loading model and tokenizer ...")
final_model = tf.keras.models.load_model(config.GENERATOR_WEIGHTS)
with open(config.FLAT_CAPTIONS_PICKLE, "rb") as fh:
    flat_train_caps = pickle.load(fh)
tokenizer = create_tokenizer(flat_train_caps)

_inception = tf.keras.applications.InceptionV3(weights="imagenet")
pt_model = tf.keras.models.Model(_inception.input, _inception.layers[-2].output)
print("Ready.")


def encode(img: np.ndarray) -> np.ndarray:
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.inception_v3.preprocess_input(x)
    fea_vec = pt_model.predict(x, verbose=0)
    return np.reshape(fea_vec, fea_vec.shape[1])


# Requests are served on separate threads (browsers keep idle connections open,
# which would block a single-threaded server); the models run one at a time.
_model_lock = threading.Lock()


def caption_for_image(img: np.ndarray) -> str:
    with _model_lock:
        picture = encode(img).reshape((1, CONFIG.feature_dim))
        return generate_caption(tokenizer, picture, final_model, CONFIG.max_length).strip()


class CaptioningHandler(BaseHTTPRequestHandler):
    def _cors_headers(self, length: int) -> None:
        self.send_header("Content-Length", str(length))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods",
                         "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")

    def do_OPTIONS(self):
        # CORS preflight: the page posts with Content-Type image/*, which
        # browsers do not send without first getting this response.
        self.send_response(204)
        self._cors_headers(0)
        self.end_headers()

    def do_GET(self):
        self._send_text(200, "Image captioning server is running.")

    def _send_text(self, status: int, text: str) -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self._cors_headers(len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        image_bytes = self.rfile.read(length)

        image_numpy = np.frombuffer(image_bytes, dtype="uint8")
        img = cv2.imdecode(image_numpy, cv2.IMREAD_COLOR) if length else None
        if img is None:
            self._send_text(400, "Could not decode the uploaded file as an image.")
            return
        # InceptionV3 expects 299x299 inputs, as used when caching training features.
        img = cv2.resize(img, CONFIG.inception_input, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            caption = caption_for_image(img)
        except Exception as exc:  # keep serving after a bad request
            self._send_text(500, f"Captioning failed: {exc}")
            return
        self._send_text(200, caption)


def main() -> None:
    server = ThreadingHTTPServer((HOST_NAME, SERVER_PORT), CaptioningHandler)
    print(f"Server started at http://{HOST_NAME}:{SERVER_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    server.server_close()
    print("Server stopped.")


if __name__ == "__main__":
    main()
