# Class-Agnostic Object Proposals

Detects candidate object regions in an image without knowing what class to look for — the same problem Selective Search and EdgeBoxes were solving before deep detectors made it look easy. This implementation uses three visual cues combined with Naive Bayes to score candidate windows.

## How it works

Three cues are computed for every candidate bounding box:

- **Multi-Scale Saliency (MS)** — spectral residual saliency computed at 5 scales {16, 24, 32, 48, 64}. Each scale votes independently; a window that's salient across scales scores higher (0–5).
- **Color Contrast (CC)** — chi-square distance between the color histogram inside the window and the surrounding ring, computed in LAB color space with 512-bin quantization. Objects tend to look different from their background.
- **Superpixel Straddling (SS)** — penalizes windows that cut through Felzenszwalb superpixel boundaries. Real objects usually align with segmentation borders.

These three scores are combined using Naive Bayes:

```
p(object | MS, CC, SS) ∝ p(MS | obj) × p(CC | obj) × p(SS | obj) × p(obj)
```

Likelihoods are learned from ~11,000 windows sampled from Pascal VOC 2012.

## Learned parameters

```
MS thresholds: {16: 0.45, 24: 0.50, 32: 0.55, 48: 0.60, 64: 0.65}
CC expansion factor (θ_cc): 1.7
SS scale (θ_ss): 150
Class prior: p(obj) = 0.045
```

## How to run

```bash
pip install numpy opencv-python scikit-image scipy pillow
python main.py
```

Results (top-scoring proposal overlaid on each image) are written to `Results/`.

## What I'd improve

The window sampling is uniform, which wastes proposals on flat background regions. A smarter strategy — like seeding windows from edges or segment boundaries — would improve recall at the same number of proposals.
