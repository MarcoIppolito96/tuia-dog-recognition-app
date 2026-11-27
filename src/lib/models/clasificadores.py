"""Arquitecturas de los clasificadores de la Etapa 2.

Las clases viven en un módulo (no en la notebook) para que `torch.load` las encuentre al cargar
el checkpoint en la app. Ambas exponen:
  - forward(x)   → logits (una salida por raza)
  - embedding(x) → vector de la capa previa al clasificador (para la búsqueda por similitud)
y guardan `razas` (índice → nombre) para traducir la predicción sin depender del dataset.
"""
from __future__ import annotations

import torch
from torch import nn


class ResNetRazas(nn.Module):
    """Modelo A: ResNet18 preentrenada en ImageNet con una capa final nueva (una salida por raza)."""

    def __init__(self, razas: list[str], preentrenada: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if preentrenada else None)
        dim = base.fc.in_features  # 512
        base.fc = nn.Identity()  # el backbone devuelve el vector del global average pooling
        self.backbone = base
        self.clasificador = nn.Linear(dim, len(razas))
        self.razas = list(razas)
        self.dim_embedding = dim

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.embedding(x))


def _bloque(entrada: int, salida: int) -> nn.Sequential:
    """Conv 3×3 → BatchNorm → ReLU, dos veces, y MaxPool que reduce la imagen a la mitad."""
    return nn.Sequential(
        nn.Conv2d(entrada, salida, 3, padding=1, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.Conv2d(salida, salida, 3, padding=1, bias=False),
        nn.BatchNorm2d(salida),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNPropia(nn.Module):
    """Modelo B: CNN diseñada desde cero.

    `canales` define los bloques convolucionales (cada uno duplica los filtros y reduce la imagen
    a la mitad). Después: global average pooling → capa de embedding (dim_embedding) → clasificador.
    """

    def __init__(
        self,
        razas: list[str],
        canales: tuple[int, ...] = (32, 64, 128, 256, 512),
        dim_embedding: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        capas, entrada = [], 3
        for salida in canales:
            capas.append(_bloque(entrada, salida))
            entrada = salida
        self.features = nn.Sequential(*capas)
        self.pool = nn.AdaptiveAvgPool2d(1)  # un número por filtro: independiente del tamaño
        self.capa_embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(entrada, dim_embedding),
            nn.BatchNorm1d(dim_embedding),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.clasificador = nn.Linear(dim_embedding, len(razas))
        self.razas = list(razas)
        self.dim_embedding = dim_embedding

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.capa_embedding(self.pool(self.features(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clasificador(self.dropout(self.embedding(x)))
