# Corruption Pipeline Rules

- NEVER pre-save corrupted images to disk. Corruptions are applied at runtime during data loading.
- Training corruptions are RANDOM (different each epoch). Validation/test corruptions are DETERMINISTIC (from manifests).
- All 4 corruption classes must appear with EQUAL probability during training.
- Salt-and-pepper: replace pixels with 0 or 1 (not 0/255 — images are normalized to [0,1]).
- Gaussian blur: use `torchvision.transforms.GaussianBlur` or `scipy.ndimage.gaussian_filter`.
- Occlusion: draw filled black rectangles on the image tensor. Coordinates are random but area constraint must be met.
- The corruption label (int 0–3) must be returned alongside corrupted/clean images for the classifier in Task 2.
