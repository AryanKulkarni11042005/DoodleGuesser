"""
Guess My Doodle — Gradio app
Run from the project root:   python app.py
Expects the trained weights at:  models/doodle_cnn.pt
"""
import os
from pathlib import Path

# Anaconda's Git Bash activation sets SSL_CERT_FILE to a path that doesn't
# exist on Windows, which makes `import gradio` crash. If it's broken, point
# it at certifi's bundle (or drop it) BEFORE gradio/httpx get imported.
if os.environ.get("SSL_CERT_FILE") and not os.path.isfile(os.environ["SSL_CERT_FILE"]):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        del os.environ["SSL_CERT_FILE"]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from PIL import Image

# ---------------------------------------------------------------------------
# Config — MUST match training exactly (same names, same ORDER = same labels)
# ---------------------------------------------------------------------------
CATEGORIES = ["airplane", "bicycle", "cat", "car", "dog",
              "fish", "house", "sun", "tree", "umbrella"]
MODEL_PATH = Path(__file__).parent / "models" / "doodle_cnn.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flip to True to see what the model gets WITHOUT proper preprocessing.
NAIVE = False


# ---------------------------------------------------------------------------
# Model — same architecture as the notebook (weights only store numbers,
# not the structure, so the class has to be redefined here)
# ---------------------------------------------------------------------------
class DoodleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = DoodleCNN(len(CATEGORIES)).to(device)
# map_location: lets weights saved on GPU load on a CPU-only machine too
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # dropout OFF -> same drawing always gives the same answer


# ---------------------------------------------------------------------------
# Preprocessing: turn a big canvas drawing into a Quick Draw-style 28x28
# ---------------------------------------------------------------------------
def to_ink(img: np.ndarray) -> np.ndarray:
    """Canvas image -> 2D uint8 array where HIGH = ink, LOW = background
    (the same convention as the training data)."""
    img = img.astype(np.float32)
    if img.ndim == 2:                      # already grayscale
        gray = img
    else:
        rgb = img[..., :3]
        if img.shape[-1] == 4:
            # Untouched canvas pixels are often TRANSPARENT with RGB = (0,0,0).
            # Converting straight to gray would make them black = same as ink!
            # So paste the drawing onto white first, using alpha as the weight.
            alpha = img[..., 3:4] / 255.0
            rgb = rgb * alpha + 255.0 * (1 - alpha)
        gray = rgb.mean(axis=-1)
    # Canvas: black ink on white. Training: white ink (255) on black (0). Invert.
    return (255.0 - gray).astype(np.uint8)


def preprocess(img: np.ndarray):
    """Returns a (1,1,28,28) float tensor, or None if the canvas is empty."""
    if NAIVE:
        pil = Image.fromarray(img).convert("L").resize((28, 28))
        x = np.array(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(x).reshape(1, 1, 28, 28)

    ink = to_ink(img)

    # 1. Find where the drawing is (ignore faint noise below 30)
    rows = np.where(ink.max(axis=1) > 30)[0]
    cols = np.where(ink.max(axis=0) > 30)[0]
    if len(rows) == 0:
        return None
    crop = ink[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]

    # 2. Pad to a SQUARE, drawing centred, with a small margin.
    #    Square first so resizing doesn't squash a wide car into a tall one.
    h, w = crop.shape
    side = int(max(h, w) * 1.15)            # ~7% margin each side, like Quick Draw
    square = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = crop

    # 3. Shrink to 28x28. LANCZOS averages neighbouring pixels -> soft edges
    #    (the anti-aliasing you saw in the training data).
    small = np.array(Image.fromarray(square).resize((28, 28), Image.LANCZOS),
                     dtype=np.float32)

    # 4. Thin canvas strokes get faint after a big shrink. Training images always
    #    peak at 255, so stretch ours to peak at 1.0 too.
    small = small / max(small.max(), 1.0)

    return torch.from_numpy(small).reshape(1, 1, 28, 28)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict(sketch):
    if sketch is None:
        return {}, None
    img = sketch["composite"] if isinstance(sketch, dict) else sketch
    if img is None:
        return {}, None

    x = preprocess(img)
    if x is None:                          # nothing drawn yet
        return {}, None

    with torch.no_grad():                  # inference: no gradients needed at all
        probs = F.softmax(model(x.to(device)), dim=1)[0].cpu()

    # What the model actually "sees", blown up 10x so you can inspect it
    view = (x.squeeze().numpy() * 255).astype(np.uint8)
    view = np.array(Image.fromarray(view).resize((280, 280), Image.NEAREST))

    return {CATEGORIES[i]: float(p) for i, p in enumerate(probs)}, view


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
# Mobile tweaks:
#  - touch-action: none -> a finger drag DRAWS instead of scrolling the page
#  - the canvas shrinks to the screen width and stays square
#  - tighter padding on small screens
CSS = """
#sketch canvas { touch-action: none; }
#sketch { max-width: 520px; margin: 0 auto; }
@media (max-width: 640px) {
    .gradio-container { padding: 8px !important; }
    #title h1 { font-size: 1.5rem; }
}
"""

with gr.Blocks(title="Guess My Doodle") as demo:
    gr.Markdown("# Guess My Doodle\nDraw one of: **" + ", ".join(CATEGORIES) + "**",
                elem_id="title")

    # Two columns side by side on desktop; min_width makes them stack
    # (canvas on top, guesses below) once the screen is narrower than ~640px.
    with gr.Row(equal_height=False):
        with gr.Column(min_width=300):
            sketch = gr.Sketchpad(
                type="numpy",
                image_mode="RGBA",
                canvas_size=(500, 500),     # square, so doodles aren't squashed
                brush=gr.Brush(default_size=10, colors=["#000000"],
                               color_mode="fixed"),
                eraser=gr.Eraser(default_size=20),
                layers=False,               # fewer toolbar buttons -> fits phones
                transforms=(),
                show_label=False,
                elem_id="sketch",
            )
        with gr.Column(min_width=300):
            guess = gr.Label(num_top_classes=3, label="My guess")
            seen = gr.Image(label="What the model sees (28x28)",
                            height=200, interactive=False)

    # Re-predict as you draw. "always_last" skips stale requests on slow
    # phone connections instead of queueing one per stroke.
    sketch.change(predict, inputs=sketch, outputs=[guess, seen],
                  trigger_mode="always_last", show_progress="hidden")

if __name__ == "__main__":
    # To try it on your phone (same Wi-Fi), use server_name="0.0.0.0" and open
    # http://<your-PC-IP>:7860 on the phone, or share=True for a public link.
    demo.launch(css=CSS, share=True)