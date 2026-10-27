"""
FastAPI service for the MNIST models

Run locally:
    uvicorn app.api:app --reload

Then open http://127.0.0.1:8000/docs
"""

import os
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from advrobust.inference import (
    SCORE_EPSILONS,
    attack,
    load_model,
    predict_proba,
    preprocess_bytes,
    robustness_score,
)

WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "app/weights")

MODELS = {
    "standard": load_model(os.path.join(WEIGHTS_DIR, "mnist_multiclass_cnn.pt")),
    "robust": load_model(os.path.join(WEIGHTS_DIR, "mnist_multiclass_cnn_adv.pt")),
}

app = FastAPI(
    title="Adversarial robustness API",
    description="MNIST digit classifier with FGSM / PGD attacks and a robustness score",
)


@app.get("/health")
def health():
    return {"status": "ok", "models": list(MODELS)}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="image of a single digit"),
    model: Literal["standard", "robust"] = "standard",
    attack_method: Literal["none", "fgsm", "pgd"] = Query("none", alias="attack"),
    epsilon: float = Query(0.1, ge=0.0, le=0.5),
):
    try:
        x = preprocess_bytes(await file.read())
    except Exception:
        raise HTTPException(status_code=400, detail="could not read the image")

    m = MODELS[model]

    clean_probs = predict_proba(m, x)
    x_adv = attack(m, x, attack_method, epsilon)
    adv_probs = predict_proba(m, x_adv)

    score = robustness_score(m, x)

    return {
        "model": model,
        "prediction": int(clean_probs.argmax()),
        "confidence": round(float(clean_probs.max()), 4),
        "attack": attack_method,
        "epsilon": epsilon,
        "prediction_after_attack": int(adv_probs.argmax()),
        "confidence_after_attack": round(float(adv_probs.max()), 4),
        "robustness_score": {
            "min_epsilon_to_flip": score,
            "searched_up_to": SCORE_EPSILONS[-1],
            "note": "smallest PGD epsilon that changes the prediction; null means it never changed",
        },
    }
