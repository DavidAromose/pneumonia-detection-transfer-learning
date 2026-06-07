# Impact of Model Complexity on Transfer Learning-Based Pneumonia Detection

A controlled comparison of three pretrained convolutional neural networks (MobileNetV2, ResNet50, EfficientNetB0) for binary pneumonia detection on chest X-rays, focused on one question: **does a bigger, more complex model actually generalise better on a small, imbalanced medical imaging dataset?**

Short answer, from this study: **no, not in any way that matters.** Once preprocessing is matched to each backbone and every model is trained to convergence, all three reach a comparable ROC-AUC (0.937–0.962). The meaningful differences are not in accuracy but in the **sensitivity/specificity trade-off** and **computational cost**.

---

## Key results

| Model | Params | GFLOPs | Latency (ms) | Test Acc | ROC-AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 2.42M | 0.61 | 297 | 83.0% | **0.962** | **0.985** | 0.573 | 0.879 |
| ResNet50 | 23.85M | 7.75 | 603 | 84.6% | 0.955 | 0.979 | 0.624 | 0.888 |
| EfficientNetB0 | 4.21M | 0.80 | 496 | **87.0%** | 0.937 | 0.890 | **0.838** | **0.895** |

95% bootstrap confidence intervals (1,000 resamples) on the 624-image test set:

- MobileNetV2 — Acc [0.803, 0.861], AUC [0.948, 0.975]
- ResNet50 — Acc [0.817, 0.873], AUC [0.938, 0.970]
- EfficientNetB0 — Acc [0.843, 0.894], AUC [0.918, 0.955]

**Takeaway:** MobileNetV2 is the best choice when sensitivity and efficiency matter most (highest recall, lowest cost). EfficientNetB0 gives the most balanced and accurate operating point. ResNet50 carries ~10x the parameters and the highest latency without a corresponding gain.

---

## The revision story

This project is also a small case study in handling peer review honestly.

An earlier version of this work was submitted to a conference and **rejected**. The reviews raised real methodological concerns: a fixed and likely insufficient training budget, missing efficiency metrics, no statistical validation, and inconsistent preprocessing across models.

Working through that feedback, I found that the original results were partly driven by **a preprocessing bug of my own**. All three models had been fed the same `1./255` rescaling, but ResNet50 and EfficientNetB0 expect their own architecture-specific preprocessing. Feeding EfficientNetB0 the wrong input distribution was the real reason it appeared to "collapse" to ~62% accuracy in the first version — not the model, the inputs.

Fixing that, and addressing the rest of the feedback, **changed the paper's conclusion**. The corrected, fairer comparison is a stronger and more defensible result than the original. What changed:

- **Per-backbone preprocessing** — each model now receives the inputs it was pretrained on, removing an unfair confound.
- **Two-phase training to convergence** — frozen-base warm-up, then selective fine-tuning of the top layers, with early stopping, replacing the fixed 10-epoch budget.
- **Consistent imbalance handling** — class weighting applied to all three models, not just one.
- **Efficiency metrics** — parameter count, FLOPs, and inference latency reported per model.
- **Statistical validation** — 95% bootstrap confidence intervals on accuracy and ROC-AUC.
- **New figures** — clean, programmatically generated diagrams (replacing an AI-generated figure the reviewers flagged), plus regenerated confusion matrices and ROC curves from the corrected run.

---

## Repository contents

| File | Description |
|---|---|
| `Pneumonia_Detection_using_Transfer_Learning.ipynb` | End-to-end notebook: data pipeline, two-phase training, evaluation, all figures. Runs top-to-bottom in one session. |
| `Pneumonia_detection_reviewed.pdf` | The revised conference paper (IEEE format). |
| `figures/` | Confusion matrices, ROC curves, training curves, and pipeline diagrams. |
| `results_summary.csv` | Final per-model metrics. |

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — Kaggle.

- 5,216 training images (1,341 Normal / 3,875 Pneumonia — roughly 1:2.9 imbalance)
- 624 test images (234 Normal / 390 Pneumonia)


---

## How to run

1. Open `Pneumonia_Detection_Consolidated.ipynb` in Google Colab (GPU runtime recommended) or Jupyter.
2. Download the dataset from the link above and set `BASE_DIR` in the configuration cell.
3. Run all cells. Outputs (`results_summary.csv`, confusion matrices, ROC curve, training curves) are saved to the working directory.

For a quick smoke test, set `EPOCHS_HEAD = 2` and `EPOCHS_FT = 1` to confirm it runs on your data layout before the full run.

**Environment:** Python 3, TensorFlow 2.x, scikit-learn, matplotlib, seaborn, pandas, numpy.

---

## Method at a glance

- **Backbones:** MobileNetV2, ResNet50, EfficientNetB0, all pretrained on ImageNet.
- **Custom head (identical across all three):** Global Average Pooling → Dense(128, ReLU) → Dropout(0.3) → Sigmoid.
- **Training:** Adam, binary cross-entropy. Phase 1 trains the head with a frozen base (lr 1e-3); Phase 2 fine-tunes the top layers (lr 1e-5). Early stopping on validation ROC-AUC; class weights throughout.
- **Augmentation:** horizontal flip, small rotation, zoom (training split only).
- **Evaluation:** accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, confusion matrix, plus FLOPs, latency, and bootstrap CIs.

---

## Limitations and future work

A single public dataset was used, which may not reflect real clinical variability; cross-dataset and multi-institutional validation is the natural next step. Fine-tuning was limited to the top layers. Future directions include stronger imbalance-mitigation strategies, hybrid and transformer-based architectures, and calibration/threshold selection so the sensitivity/specificity trade-off can be tuned to a specific clinical setting.

---

## Citation

If you reference this work, please cite the paper (see `pneumonia_detection_reviewed.pdf`).

> D. Aromose, S. E. Uwah, and C. Iwendi, "Impact of Model Complexity on Transfer Learning-Based Pneumonia Detection Using Public Chest X-Ray Images."

---

## Author

**David Aromose** — MRes in Applied Artificial Intelligence, University of Greater Manchester
ORCID: [0009-0008-7428-6650](https://orcid.org/0009-0008-7428-6650)

*Research interests: AI Engineering, Software Engineering, AI safety and security, MLSecOps, responsible AI.*
