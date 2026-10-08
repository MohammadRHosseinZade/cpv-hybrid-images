import numpy as np

from hybrid import (
    _gaussian_blur_separable,
    convolve_2d,
    cross_correlation_2d,
    gaussian_blur_kernel_2d,
    high_pass,
    low_pass,
    make_hybrid,
    mouth_mask_from_landmarks,
)


def test_cross_correlation_2d_uses_zero_padding() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])

    result = cross_correlation_2d(image, kernel)

    assert np.array_equal(result, np.array([[-1.0, -2.0], [-3.0, -3.0]]))


def test_convolve_2d_flips_kernel_before_correlation() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    kernel = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = convolve_2d(image, kernel)

    assert np.array_equal(result, np.array([[1.0, 4.0], [6.0, 20.0]]))


def test_gaussian_kernel_is_normalized_and_symmetric() -> None:
    kernel = gaussian_blur_kernel_2d(3, 3, 1.0)

    assert np.isclose(kernel.sum(), 1.0)
    assert np.isclose(kernel[0, 0], kernel[2, 2])
    assert kernel[1, 1] > kernel[0, 0]


def test_separable_gaussian_blur_matches_2d_convolution() -> None:
    image = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    expected = convolve_2d(image, gaussian_blur_kernel_2d(5, 5, 1.0))

    result = _gaussian_blur_separable(image, kernel_size=5, sigma=1.0)

    assert np.allclose(result, expected)


def test_mouth_mask_peaks_beyond_nose_from_eye_midpoint() -> None:
    mask = mouth_mask_from_landmarks(
        (1000, 1000, 3),
        left_eye=(600.0, 300.0),
        right_eye=(600.0, 700.0),
        nose=(400.0, 500.0),
    )

    peak_y, peak_x = np.unravel_index(np.argmax(mask), mask.shape)
    assert mask.shape == (1000, 1000)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    assert abs(peak_x - 112) <= 1
    assert abs(peak_y - 500) <= 1


def test_low_pass_blurs_impulse() -> None:
    image = np.zeros((3, 3))
    image[1, 1] = 1.0

    result = low_pass(image, kernel_size=3, sigma=1.0)

    assert result[1, 1] < 1.0
    assert result[0, 0] > 0.0


def test_high_pass_is_original_minus_low_pass() -> None:
    image = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = high_pass(image, kernel_size=3, sigma=1.0)

    assert np.allclose(result, image - low_pass(image, kernel_size=3, sigma=1.0))


def test_make_hybrid_uses_weighted_low_and_high_layers() -> None:
    low_source = np.full((3, 3), 10.0)
    high_source = np.full((3, 3), 20.0)

    result = make_hybrid(
        low_source,
        high_source,
        low_kernel_size=3,
        low_sigma=1.0,
        low_weight=0.8,
        high_kernel_size=3,
        high_sigma=1.0,
        high_weight=1.2,
    )

    expected = 0.8 * low_pass(low_source, 3, 1.0) + 1.2 * high_pass(
        high_source, 3, 1.0
    )
    assert np.allclose(result, expected)


def test_make_hybrid_with_zero_mask_uses_only_low_layer() -> None:
    source = np.full((3, 3), 10.0)

    result = make_hybrid(
        source,
        np.full((3, 3), 20.0),
        low_kernel_size=3,
        low_sigma=1.0,
        low_weight=0.8,
        high_kernel_size=3,
        high_sigma=1.0,
        high_weight=1.2,
        high_mask=np.zeros((3, 3)),
    )

    assert np.allclose(result, 0.8 * low_pass(source, 3, 1.0))
