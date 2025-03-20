import kornia as K


def test_noise(image):
    """Тест уровня шума."""
    image_tensor = K.image_to_tensor(image, keepdim=False).float() / 255.0
    blurred = K.filters.gaussian_blur2d(
        image_tensor, kernel_size=(5, 5), sigma=(1.5, 1.5)
    )
    noise = (image_tensor - blurred).abs().mean().item()
    return noise
