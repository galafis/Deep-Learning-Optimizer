#!/usr/bin/env python3
"""
Deep-Learning-Optimizer API
Flask API for benchmarking and comparing gradient-based optimizers.
Author: Gabriel Demetrios Lafis
"""

from flask import Flask, jsonify, request
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from optimizers import SGD, MomentumSGD, AdaGrad, RMSProp, Adam, AdamW, benchmark_optimizers

app = Flask(__name__)


def quadratic(params):
    return 0.5 * np.sum(params ** 2)

def quadratic_grad(params):
    return params.copy()


@app.route("/")
def index():
    return jsonify({
        "project": "Deep-Learning-Optimizer",
        "description": "Gradient-based optimizer benchmarking API",
        "author": "Gabriel Demetrios Lafis",
        "endpoints": ["/api/status", "/api/optimizers", "/api/benchmark"],
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/status")
def status():
    return jsonify({"status": "running", "version": "1.0.0"})


@app.route("/api/optimizers")
def list_optimizers():
    return jsonify({
        "optimizers": [
            {"name": "sgd", "description": "Stochastic Gradient Descent"},
            {"name": "momentum", "description": "SGD with Momentum"},
            {"name": "adagrad", "description": "Adaptive Gradient"},
            {"name": "rmsprop", "description": "Root Mean Square Propagation"},
            {"name": "adam", "description": "Adaptive Moment Estimation"},
            {"name": "adamw", "description": "Adam with Decoupled Weight Decay + Cosine Annealing"},
        ]
    })


@app.route("/api/benchmark", methods=["POST"])
def run_benchmark():
    data = request.json or {}
    n_steps = data.get("n_steps", 100)
    dimension = data.get("dimension", 5)
    lr = data.get("learning_rate", 0.01)

    np.random.seed(42)
    initial = np.random.randn(dimension) * 5.0

    optimizers = {
        "sgd": SGD(lr),
        "momentum": MomentumSGD(lr, beta=0.9),
        "adagrad": AdaGrad(lr * 10),
        "rmsprop": RMSProp(lr),
        "adam": Adam(lr),
        "adamw": AdamW(lr, weight_decay=0.01, cosine_annealing=True, T_max=n_steps),
    }

    results = benchmark_optimizers(
        quadratic, quadratic_grad, initial, optimizers, n_steps=n_steps
    )

    summary = {}
    for name, losses in results.items():
        summary[name] = {
            "initial_loss": round(losses[0], 6),
            "final_loss": round(losses[-1], 6),
            "improvement_pct": round((1 - losses[-1] / losses[0]) * 100, 2) if losses[0] > 0 else 0,
            "steps": len(losses),
        }

    return jsonify({"benchmark": summary, "dimension": dimension, "n_steps": n_steps})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
