# VACE: Training & Validation Flowchart, and Experiment Plan (English)

This document provides two parts: (A) **Training + Validation flowcharts** for integrating VACE loss; (B) **Experiments to run** and **data to collect** for a WACV-style submission.

---

## A) Training + Validation Flowcharts

### A.1 Training Flow (with VACE)

```mermaid
flowchart TD
  A[Start / Load Config] --> B[Create Datasets & Dataloaders]
  B --> C[Build Backbone (timm CNN)]
  C --> D[Build Classifier Head (Linear)]
  D --> E[Instantiate VACE Loss (a,b,tau_min,max, ema_decay)]
  E --> F[Choose Optimizer & LR Scheduler]
  F --> G[for epoch in 1..E]
  G --> H[for each batch]
  H --> I[Load batch (x, y)]
  I --> J[Forward features: f = model.forward_features(x)]
  J --> K[Logits: z = W^T f (+ b)]
  K --> L[VACE: compute τ from EMAStats; scale logits z/τ]
  L --> M[Loss = CE(z/τ, y)]
  M --> N[Backward: dθ = ∂Loss/∂θ]
  N --> O[Optimizer.step(); Scheduler.step_if_needed]
  O --> P[Update EMAStats with batch (no grad)]
  P --> Q[Log train metrics, τ distribution]
  Q --> R{End of epoch?}
  R -- No --> H
  R -- Yes --> S[Run Validation]
  S --> T[Checkpoint (best by val metric)]
  T --> U[Next epoch]
  U --> V[End]
```

**Notes:**

* Mixed precision (AMP) and gradient accumulation are optional but recommended for speed/VRAM.
* `EMAStats.update()` is called **only in training mode**.
* `τ = clip(a + b·σ², τ_min, τ_max)`, computed per-class from EMA-smoothed per-class logit variance.
* Checkpointing: save `state_dict` of backbone, head, and VACE buffers (including EMAStats and current τ).

### A.2 Validation / Test Flow (no stats update)

```mermaid
flowchart TD
  A[Eval Mode (model.eval(); no_grad)] --> B[for each val batch]
  B --> C[Forward features f = model.forward_features(x)]
  C --> D[Logits z = W^T f (+ b)]
  D --> E[Freeze τ from current VACE buffers]
  E --> F[Compute Loss = CE(z/τ, y); DO NOT update EMAStats]
  F --> G[Collect predictions, probs, confidences]
  G --> H[Aggregate metrics (Top-1, ECE, NLL, etc.)]
  H --> I[End]
```

**Notes:**

* Do **not** call `EMAStats.update()` in eval; τ stays fixed for the whole evaluation pass.
* For calibration metrics, compute probabilities with `softmax(z/τ)`.

---