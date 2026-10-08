# Hybrid Image Design

## Goal

Build a self-contained Python implementation that combines the aligned neutral and smiling portraits into one hybrid image. Close viewing reveals the smiling expression; distant or reduced viewing reveals the neutral expression.

## Inputs and outputs

- Input `aligned/left.png`: neutral portrait. It supplies the low-frequency layer.
- Input `aligned/right.png`: smiling portrait. It supplies the high-frequency layer.
- Output `hybrid.png`: merged image, in RGB PNG format and the same dimensions as the aligned inputs.

## Architecture

`hybrid.py` owns all required filtering logic and a command-line runner. It loads both images with Pillow, converts them to floating-point NumPy arrays, filters each RGB channel with project-owned functions, combines layers, clips values to the 0–255 range, and saves a PNG.

The implementation contains exactly these required public functions:

1. `cross_correlation_2d(image, kernel)`
2. `convolve_2d(image, kernel)`
3. `gaussian_blur_kernel_2d(height, width, sigma)`
4. `low_pass(image, kernel_size, sigma)`
5. `high_pass(image, kernel_size, sigma)`

`convolve_2d` flips its kernel vertically and horizontally before calling `cross_correlation_2d`. Gaussian blur uses a normalized two-dimensional Gaussian kernel. Low-pass filtering convolves the image with this Gaussian. High-pass filtering subtracts that low-pass result from the original image.

## Border and color behavior

Images use zero padding at borders. Filtering accepts either a grayscale 2D image or RGB 3D image; RGB filtering runs independently per channel. Calculations use `float64`, avoiding unsigned-byte subtraction errors. Output conversion happens only after final clipping.

## Hybrid combination

The runner calculates:

```text
neutral_low = low_pass(left, low_kernel_size, low_sigma)
smile_high = high_pass(right, high_kernel_size, high_sigma)
hybrid = low_weight * neutral_low + high_weight * smile_high
```

Default filter values will be selected during visual inspection, then documented in `README.md`. The runner exposes each filter’s kernel size, sigma, and layer weight as command-line options so parameters can be tuned without changing code.

## Verification

`tests/test_hybrid.py` uses tiny known arrays to verify cross-correlation, kernel flipping in convolution, Gaussian normalization and symmetry, low-pass blur behavior, and high-pass subtraction. Tests run with pytest. Final visual validation checks that `hybrid.png` opens, is RGB, matches input dimensions, and exhibits intended close/far interpretations.

## Constraints

- Do not call ready-made filtering or convolution functions from NumPy, SciPy, OpenCV, Pillow, or other libraries.
- Use only allowed base NumPy operations and project-owned loops/vectorization for filtering.
- Preserve provided `gui.py`; it remains alignment helper only.
