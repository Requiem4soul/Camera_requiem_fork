import kornia as K


def test_sharpness(image):
    """Тест резкости изображения."""
    image_tensor = K.image_to_tensor(image, keepdim=False).float() / 255.0
    grads = K.filters.spatial_gradient(image_tensor)
    sharpness = grads.abs().sum(dim=1).mean().item()
    return sharpness
