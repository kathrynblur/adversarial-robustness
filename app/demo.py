"""
Gradio demo: draw a digit, attack it with FGSM or PGD, and compare
a standard CNN with an adversarially trained CNN

Run locally:
    python app/demo.py
"""

import os

import gradio as gr
import numpy as np
import torch
from PIL import Image

from advrobust.inference import attack, load_model, predict_proba, preprocess, robustness_score

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", os.path.join(APP_DIR, "weights"))

MODELS = {
    "Standard CNN": load_model(os.path.join(WEIGHTS_DIR, "mnist_multiclass_cnn.pt")),
    "Adversarially trained CNN": load_model(os.path.join(WEIGHTS_DIR, "mnist_multiclass_cnn_adv.pt")),
}

# hide the brush color, brush size and eraser buttons: the brush is always black and the same size,
# and the trash button (clear canvas) is the only way to start over.
# Also hide pan, zoom and the zoom level, which a small digit canvas does not need
CSS = (
    "#digit button[aria-label='Color'], "
    "#digit button[aria-label='Brush Size'], "
    "#digit button[aria-label='Erase'], "
    "#digit button[aria-label='Pan'], "
    "#digit button[aria-label='Zoom out'], "
    "#digit button[aria-label='Zoom in'], "
    "#digit .zoom-number, "
    "#digit .zoom-number + .separator {display: none !important;}"
)

EXPLANATION = """
### About this project
In my bachelor thesis, *A Theoretical and Empirical Study of Adversarial Vulnerabilities in Neural Networks*,
I studied a strange weakness of neural networks: you can apply small calculated changes to an image so that
a human sees no difference, but the network still gives a completely different answer.
I analyzed this both mathematically and in experiments with logistic regression, CNNs and a pretrained ResNet-18
on MNIST and STL-10. This demo lets you reproduce and test this anomaly on drawn digits.

### What is an adversarial example?
An *adversarial example* is an image with tiny, carefully chosen changes. To a human, it still shows the same digit,
but the network predicts a different one.

### What you are looking at
You draw a digit. Two neural networks try to recognize it, and an attack then tries to fool them.
Both networks are the same small convolutional neural network (CNN), trained on 70,000 handwritten MNIST digits,
and both reach about 99% accuracy on normal test images. The difference is in the training:
- the **standard CNN** only ever saw normal images,
- the **adversarially trained CNN** was trained on a 50/50 mix of normal and FGSM-attacked images (epsilon = 0.15).

### How the attacks work
Both attacks use the network's own gradient: they ask the network which pixel changes would hurt it the most.
- **FGSM (Fast Gradient Sign Method)** takes one single step in that direction. It is fast, but weak.
- **PGD (Projected Gradient Descent)** takes 20 small steps and keeps every pixel within the allowed range.
  It is much stronger, and it is the standard way to test whether a model is really robust.

**Epsilon** sets the largest change allowed for each pixel, on a scale from 0 (black) to 1 (white).
The **robustness score** is the smallest epsilon at which PGD changes the network's answer for your digit.
Higher means more robust.

### What I found
- **Small changes are enough.** Under PGD with epsilon = 0.15, the standard CNN drops from 99% to about 23% accuracy.
- **Adversarial training works, up to a limit.** At the same epsilon, the adversarially trained CNN keeps about 90%.
  Above the epsilon it was trained with, PGD breaks it too.
- **FGSM alone is misleading.** When I tested my thesis models again with PGD, I found that FGSM had made them look
  more robust than they were. At epsilon = 0.25, FGSM suggests 84% accuracy for the robust model,
  but PGD shows the true worst case: about 20%.

### What I built
After the thesis, I turned the research code into a Python package with tests and continuous integration,
reproducible experiments from config files, experiment tracking with Weights & Biases, a FastAPI service in Docker,
and this demo.
"""


def to_display(x, size=196):
    """
    Turn a (1, 1, 28, 28) tensor into an enlarged grayscale image
    """
    array = (x[0, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array).resize((size, size), Image.NEAREST)


def probabilities(probs):
    return {str(i): float(p) for i, p in enumerate(probs)}


def get_image(editor_value):
    """
    The image editor returns a dict; the 'composite' entry is the final image
    """
    if editor_value is None:
        return None
    image = editor_value.get("composite") if isinstance(editor_value, dict) else editor_value
    if image is None:
        return None
    if image.mode in ("RGBA", "LA"):
        # paste onto a white background, so transparent pixels count as paper
        background = Image.new("RGBA", image.size, "white")
        image = Image.alpha_composite(background, image.convert("RGBA"))
    return image


def verdict_html(clean_pred, adv_pred):
    """
    Big 'original -> attacked' prediction with a fooled / not fooled badge
    """
    fooled = clean_pred != adv_pred
    color = "#d93025" if fooled else "#1e8e3e"
    badge = "✗ Fooled" if fooled else "✓ Not fooled"
    return f"""
    <div style="text-align: center; padding: 8px 0;">
        <div style="font-size: 2.6em; font-weight: 700;">{clean_pred} &rarr; {adv_pred}</div>
        <div style="display: inline-block; margin-top: 6px; padding: 4px 14px; border-radius: 999px;
                    background: {color}; color: white; font-weight: 600;">{badge}</div>
    </div>
    """


def score_text(score):
    if score is None:
        return "**Robustness score:** above 0.30 (the prediction survived every PGD epsilon tested)"
    return f"**Robustness score:** {score:.2f}"


def run(editor_value, method, epsilon):
    image = get_image(editor_value)
    if image is None:
        raise gr.Error("Draw a digit first.")

    x = preprocess(image)
    if x.max() == 0:
        raise gr.Error("The canvas is empty. Draw a digit first.")

    outputs = [to_display(x)]

    for model in MODELS.values():
        torch.manual_seed(0)
        x_adv = attack(model, x, method.lower(), epsilon)
        clean_probs = predict_proba(model, x)
        adv_probs = predict_proba(model, x_adv)
        outputs += [
            verdict_html(int(clean_probs.argmax()), int(adv_probs.argmax())),
            to_display(x_adv),
            score_text(robustness_score(model, x)),
            probabilities(clean_probs),
            probabilities(adv_probs),
        ]

    return outputs


with gr.Blocks(title="Adversarial examples on MNIST") as demo:
    gr.Markdown(
        """
        # Adversarial examples on MNIST
        Draw a digit, choose an attack and its strength epsilon,
        and see whether a standard CNN and an adversarially trained CNN still recognize it.
        """
    )

    with gr.Tab("Try it"):
        with gr.Row():
            with gr.Column(scale=1):
                editor = gr.ImageEditor(
                    label="Your digit",
                    elem_id="digit",
                    type="pil",
                    image_mode="RGBA",
                    sources=[],
                    transforms=[],
                    layers=False,
                    canvas_size=(280, 280),
                    brush=gr.Brush(default_size=10, colors=["#000000"], color_mode="fixed"),
                    eraser=gr.Eraser(default_size=10),
                )

            verdicts, adv_images, scores = [], [], []
            for name in MODELS:
                with gr.Column(scale=1):
                    gr.Markdown(f"### {name}")
                    verdicts.append(gr.HTML())
                    adv_images.append(gr.Image(label="Attacked image", interactive=False, height=220))
                    scores.append(gr.Markdown())

        with gr.Row():
            method = gr.Radio(["PGD", "FGSM"], value="PGD", label="Attack")
            epsilon = gr.Slider(0.0, 0.3, value=0.15, step=0.01, label="Epsilon (max change per pixel)")
        button = gr.Button("Attack", variant="primary")

        with gr.Accordion("Show details", open=False):
            with gr.Row():
                model_input = gr.Image(label="What the models see (28 x 28)", interactive=False, height=220)
                details = []
                for name in MODELS:
                    with gr.Column():
                        gr.Markdown(f"**{name}**")
                        details.append(gr.Label(num_top_classes=3, label="Prediction on the original"))
                        details.append(gr.Label(num_top_classes=3, label="Prediction on the attacked image"))

    with gr.Tab("How it works"):
        gr.Markdown(EXPLANATION)

    gr.Markdown("[GitHub adversarial robustness](https://github.com/kathrynblur/adversarial-robustness)")

    outputs = [model_input]
    for i in range(len(MODELS)):
        outputs += [verdicts[i], adv_images[i], scores[i], details[2 * i], details[2 * i + 1]]

    button.click(run, inputs=[editor, method, epsilon], outputs=outputs)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Ocean())
