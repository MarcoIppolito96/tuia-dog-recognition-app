"""Preprocesamiento compartido entre las notebooks y los servicios de la app.

Toda imagen entra en BGR uint8 (como la devuelven cv2.imread y la app) y sale como
tensor float32 (3, IMAGE_SIZE, IMAGE_SIZE) normalizado con las estadísticas de ImageNet.
Usar siempre estas funciones (entrenamiento, evaluación e inferencia) garantiza que el
modelo reciba las imágenes exactamente igual en la notebook y en la app.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

# Estadísticas de ImageNet (RGB, escala 0-1)
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIO_IMAGENET = (0.229, 0.224, 0.225)
# Relleno = media de ImageNet en 0-255: después de normalizar queda en ~0 (neutro)
COLOR_RELLENO = tuple(int(round(m * 255)) for m in MEDIA_IMAGENET)


def leer_imagen(ruta: str | Path) -> np.ndarray:
    """Lee una imagen del disco en BGR uint8 (misma convención que la app)."""
    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    return imagen


def a_rgb_cuadrada(imagen_bgr: np.ndarray, image_size: int) -> np.ndarray:
    """BGR → RGB, relleno hasta hacerla cuadrada (sin deformar) y resize a image_size."""
    if imagen_bgr is None or imagen_bgr.size == 0:
        raise ValueError("Imagen vacía")
    # Acepta también escala de grises o BGRA
    if imagen_bgr.ndim == 2:
        imagen_bgr = cv2.cvtColor(imagen_bgr, cv2.COLOR_GRAY2BGR)
    elif imagen_bgr.shape[2] == 4:
        imagen_bgr = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

    # Relleno centrado: un recorte alargado de YOLO conserva sus proporciones
    alto, ancho = rgb.shape[:2]
    lado = max(alto, ancho)
    arriba, izquierda = (lado - alto) // 2, (lado - ancho) // 2
    cuadrada = cv2.copyMakeBorder(
        rgb, arriba, lado - alto - arriba, izquierda, lado - ancho - izquierda,
        cv2.BORDER_CONSTANT, value=COLOR_RELLENO,
    )
    # INTER_AREA al achicar (evita aliasing), INTER_LINEAR al agrandar
    interpolacion = cv2.INTER_AREA if lado > image_size else cv2.INTER_LINEAR
    return cv2.resize(cuadrada, (image_size, image_size), interpolation=interpolacion)


def a_tensor(imagen_rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 (H, W, 3) → tensor float32 (3, H, W) normalizado con ImageNet."""
    x = imagen_rgb.astype(np.float32) / 255.0
    x = (x - np.array(MEDIA_IMAGENET, dtype=np.float32)) / np.array(DESVIO_IMAGENET, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def preprocesar(
    imagen_bgr: np.ndarray,
    image_size: int,
    aumentacion: Optional[Callable[..., dict]] = None,
) -> torch.Tensor:
    """Pipeline completo: BGR uint8 → tensor listo para el modelo.

    `aumentacion` (opcional, solo en entrenamiento) es un pipeline de albumentations que
    se aplica sobre la imagen RGB uint8 ya cuadrada, antes de normalizar.
    """
    rgb = a_rgb_cuadrada(imagen_bgr, image_size)
    if aumentacion is not None:
        rgb = aumentacion(image=rgb)["image"]
    return a_tensor(rgb)


def aumentacion_entrenamiento(image_size: int):
    """Data augmentation para train (nunca para valid, test ni el conjunto independiente).

    Se aplica sobre la imagen RGB uint8 ya cuadrada (ver `preprocesar`). Los rangos son
    moderados para no cambiar la raza aparente: sin flips verticales, sin rotaciones de 90°
    y sin cambios de tono que alteren el color del pelaje.
    """
    import albumentations as A  # solo hace falta al entrenar

    return A.Compose([
        # Encuadre: simula recortes de YOLO más o menos ajustados al perro
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.8, 1.25), p=0.5),
        A.HorizontalFlip(p=0.5),
        # Inclinación leve de cámara; el borde se rellena igual que en `a_rgb_cuadrada`
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=COLOR_RELLENO, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
    ])


def desnormalizar(tensor: torch.Tensor) -> np.ndarray:
    """Inversa de a_tensor, para visualizar: tensor (3, H, W) → RGB uint8 (H, W, 3)."""
    x = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(DESVIO_IMAGENET) + np.array(MEDIA_IMAGENET)
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)
