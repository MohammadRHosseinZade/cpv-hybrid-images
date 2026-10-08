# CPV Hybrid Images

## Result

- `aligned/left.png`: neutral portrait; source of low-frequency content.
- `aligned/right.png`: smiling portrait; source of high-frequency content.
- `hybrid.png`: final hybrid image. Close viewing emphasizes mouth and smiling-image details; distant or reduced viewing emphasizes neutral-image structure.

## Parameters

- Low-pass: 601×601 Gaussian kernel, sigma 100, weight 0.95.
- High-pass: 73×73 Gaussian kernel, sigma 12, weight 2.5.
- The high-pass layer uses a smooth mouth-region mask derived from two eye points and a nose point in `aligned/correspondence.json`.
- Processing resolution: original aligned input resolution, 6528×4896.

## Generate

```bash
uv run python hybrid.py aligned/left.png aligned/right.png --correspondence aligned/correspondence.json --output hybrid.png
```
