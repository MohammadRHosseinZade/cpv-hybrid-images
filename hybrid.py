"""Custom image filters and hybrid-image generator for CPV project 1."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def cross_correlation_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D kernel without flipping it, using zero padding."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if image.ndim not in (2, 3) or kernel.ndim != 2:
        raise ValueError("image must be 2D or 3D and kernel must be 2D")
    if kernel.shape[0] == 0 or kernel.shape[1] == 0:
        raise ValueError("kernel dimensions must be nonzero")

    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padding = ((pad_y, pad_y), (pad_x, pad_x))
    if image.ndim == 3:
        padding += ((0, 0),)
    padded = np.pad(image, padding, mode="constant")
    result = np.zeros_like(image, dtype=np.float64)

    height, width = image.shape[:2]
    for kernel_y in range(kernel.shape[0]):
        for kernel_x in range(kernel.shape[1]):
            shifted = padded[kernel_y : kernel_y + height, kernel_x : kernel_x + width]
            result += kernel[kernel_y, kernel_x] * shifted

    return result


def convolve_2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution using the cross-correlation implementation."""
    return cross_correlation_2d(image, np.flip(kernel, axis=(0, 1)))


def gaussian_blur_kernel_2d(height: int, width: int, sigma: float) -> np.ndarray:
    """Return a normalized, centered two-dimensional Gaussian kernel."""
    if height <= 0 or width <= 0 or sigma <= 0:
        raise ValueError("height, width, and sigma must be positive")

    y, x = np.mgrid[:height, :width]
    y = y - (height - 1) / 2
    x = x - (width - 1) / 2
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> np.ndarray:
    """Return a normalized, centered one-dimensional Gaussian kernel."""
    if kernel_size <= 0 or kernel_size % 2 == 0 or sigma <= 0:
        raise ValueError("kernel_size must be positive and odd, and sigma positive")
    positions = np.arange(kernel_size, dtype=np.float64) - kernel_size // 2
    kernel = np.exp(-(positions * positions) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _correlate_1d(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Apply a one-dimensional kernel along image rows or columns with zero padding."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim not in (2, 3) or axis not in (0, 1):
        raise ValueError("image must be 2D or 3D and axis must be 0 or 1")

    pad = kernel.size // 2
    padding = [(0, 0)] * image.ndim
    padding[axis] = (pad, pad)
    padded = np.pad(image, padding, mode="constant")
    result = np.zeros_like(image, dtype=np.float64)
    for index, weight in enumerate(kernel):
        source = [slice(None)] * image.ndim
        source[axis] = slice(index, index + image.shape[axis])
        result += weight * padded[tuple(source)]
    return result


def _gaussian_blur_separable(
    image: np.ndarray, kernel_size: int, sigma: float
) -> np.ndarray:
    """Blur an image with two custom one-dimensional Gaussian passes."""
    kernel = _gaussian_kernel_1d(kernel_size, sigma)
    horizontal = _correlate_1d(image, kernel, axis=1)
    return _correlate_1d(horizontal, kernel, axis=0)


def low_pass(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Keep low frequencies by convolving with a Gaussian kernel."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    return _gaussian_blur_separable(np.asarray(image, dtype=np.float64), kernel_size, sigma)


def high_pass(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Keep high frequencies by subtracting the Gaussian-blurred image."""
    image = np.asarray(image, dtype=np.float64)
    return image - low_pass(image, kernel_size, sigma)


def mouth_mask_from_landmarks(
    image_shape: tuple[int, ...],
    left_eye: tuple[float, float],
    right_eye: tuple[float, float],
    nose: tuple[float, float],
) -> np.ndarray:
    """Return a soft mouth-region mask derived from two eyes and a nose point."""
    height, width = image_shape[:2]
    left_eye_array = np.asarray(left_eye, dtype=np.float64)
    right_eye_array = np.asarray(right_eye, dtype=np.float64)
    nose_array = np.asarray(nose, dtype=np.float64)
    eye_midpoint = (left_eye_array + right_eye_array) / 2
    direction = nose_array - eye_midpoint
    direction_length = np.linalg.norm(direction)
    eye_distance = np.linalg.norm(right_eye_array - left_eye_array)
    if direction_length == 0 or eye_distance == 0:
        raise ValueError("eyes and nose must define a face direction")

    forward = direction / direction_length
    sideways = np.array([-forward[1], forward[0]])
    center = nose_array + 0.72 * eye_distance * forward
    y, x = np.ogrid[:height, :width]
    forward_distance = (x - center[0]) * forward[0] + (y - center[1]) * forward[1]
    sideways_distance = (x - center[0]) * sideways[0] + (y - center[1]) * sideways[1]
    return np.exp(
        -0.5 * ((forward_distance / 300) ** 2 + (sideways_distance / 500) ** 2)
    )


def make_hybrid(
    low_source: np.ndarray,
    high_source: np.ndarray,
    low_kernel_size: int,
    low_sigma: float,
    low_weight: float,
    high_kernel_size: int,
    high_sigma: float,
    high_weight: float,
    high_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Combine a low-frequency image layer with a high-frequency image layer."""
    low_source = np.asarray(low_source, dtype=np.float64)
    high_source = np.asarray(high_source, dtype=np.float64)
    if low_source.shape != high_source.shape:
        raise ValueError("input images must have identical dimensions")
    high_layer = high_pass(high_source, high_kernel_size, high_sigma)
    if high_mask is not None:
        high_mask = np.asarray(high_mask, dtype=np.float64)
        if high_mask.shape != low_source.shape[:2]:
            raise ValueError("high_mask must match image height and width")
        if high_layer.ndim == 3:
            high_layer *= high_mask[..., np.newaxis]
        else:
            high_layer *= high_mask
    return low_weight * low_pass(low_source, low_kernel_size, low_sigma) + high_weight * high_layer


def parse_args() -> argparse.Namespace:
    """Read input paths and filter parameters from command-line arguments."""
    parser = argparse.ArgumentParser(description="Create a hybrid image.")
    parser.add_argument("low_image", help="neutral image; supplies low frequencies")
    parser.add_argument("high_image", help="smiling image; supplies high frequencies")
    parser.add_argument("--output", default="hybrid.png", help="output PNG path")
    parser.add_argument(
        "--correspondence",
        help="JSON with eye, eye, and nose points; limits high frequencies to mouth region",
    )
    parser.add_argument("--low-kernel-size", type=int, default=601)
    parser.add_argument("--low-sigma", type=float, default=100.0)
    parser.add_argument("--low-weight", type=float, default=0.95)
    parser.add_argument("--high-kernel-size", type=int, default=73)
    parser.add_argument("--high-sigma", type=float, default=12.0)
    parser.add_argument("--high-weight", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    """Load aligned images, generate hybrid result, and save it as a PNG."""
    args = parse_args()
    low_image = np.asarray(Image.open(args.low_image).convert("RGB"), dtype=np.float64)
    high_image = np.asarray(Image.open(args.high_image).convert("RGB"), dtype=np.float64)
    high_mask = None
    if args.correspondence:
        with open(args.correspondence, encoding="utf-8") as correspondence_file:
            points = np.asarray(json.load(correspondence_file)["points1"], dtype=np.float64)
        if points.shape != (3, 2):
            raise ValueError("correspondence points1 must contain exactly two eyes and one nose")
        high_mask = mouth_mask_from_landmarks(
            low_image.shape,
            tuple(points[0]),
            tuple(points[1]),
            tuple(points[2]),
        )
    result = make_hybrid(
        low_image,
        high_image,
        args.low_kernel_size,
        args.low_sigma,
        args.low_weight,
        args.high_kernel_size,
        args.high_sigma,
        args.high_weight,
        high_mask,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(result, 0, 255).astype(np.uint8)).save(output_path)
    print(f"Saved hybrid image: {output_path}")


if __name__ == "__main__":
    main()
