from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

from lib.preprocesamiento import preprocesar
from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el Pull Request.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
        data_root: Optional[Path] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver
        # DATA_PATH: las rutas de la galeria se guardan relativas a esta carpeta para que
        # la base indexada desde el entorno local sirva tambien en Docker (y viceversa).
        self.data_root = data_root
        # Modelo de embeddings (baseline): se carga una sola vez, en la primera llamada
        self._modelo = None

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def _cargar_modelo(self):
        """ResNet18 preentrenada en ImageNet sin la capa de clasificación (salida de 512)."""
        if self._modelo is None:
            import torch
            from torchvision.models import ResNet18_Weights, resnet18

            modelo = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            modelo.fc = torch.nn.Identity()  # queda el vector del global average pooling
            self._modelo = modelo.eval()
            logger.info("Modelo de embeddings baseline cargado (ResNet18 ImageNet)")
        return self._modelo

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Embedding de una imagen BGR uint8 con ResNet18 preentrenada en ImageNet
        (baseline). Usa el preprocesamiento compartido (BGR→RGB, relleno a cuadrado,
        resize a image_size y normalización ImageNet) y devuelve 512 floats con norma L2 = 1.
        """
        import torch

        modelo = self._cargar_modelo()
        x = preprocesar(image, self.image_size).unsqueeze(0)  # dimensión de batch
        with torch.no_grad():
            emb = modelo(x)[0].numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-12)  # normalización L2
        return emb.astype(float).tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Top-k imágenes más parecidas de la base, ordenadas por score descendente.

        - pgvector: la base ordena por distancia coseno (operador <=>). Como los embeddings
          tienen norma 1, ordenar por coseno o por distancia L2 da el mismo ranking
          (||a - b||² = 2 - 2·cos), así que sirve para ambas métricas.
        - JSON: se recorren todos los registros.
        En ambos casos el score se calcula con self.similarity (SIMILARITY_METRIC).
        """
        if top_k <= 0:
            return []
        if hasattr(self.store, "search"):
            registros = self.store.search(embedding, top_k)
        else:
            registros = self.store.all()

        vecinos = [
            Neighbor(path=r.path, breed=r.breed, score=self.similarity(embedding, r.embedding))
            for r in registros
        ]
        vecinos.sort(key=lambda v: v.score, reverse=True)
        return vecinos[:top_k]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Voto de los vecinos ponderado por similitud: gana la raza con mayor suma de scores.

        El score devuelto es la **similitud media de los vecinos de la raza ganadora**
        (qué tan parecidas son, en promedio, las imágenes que votaron por ella).
        Si no hay vecinos o ese score queda por debajo de SIMILARITY_THRESHOLD, devuelve
        ("unknown", score).
        """
        if not results:
            return "unknown", 0.0

        votos: dict[str, list[float]] = {}
        for vecino in results:
            votos.setdefault(vecino.breed, []).append(vecino.score)
        # Desempate: mayor suma y, si empatan, el vecino más parecido
        raza = max(votos, key=lambda r: (sum(votos[r]), max(votos[r])))
        score = float(np.mean(votos[raza]))

        if score < self.similarity_threshold:
            return "unknown", score
        return raza, score

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def _to_stored_path(self, image_path: str) -> str:
        """Ruta relativa a DATA_PATH si la imagen esta dentro; si no, la ruta tal cual."""
        if self.data_root is None:
            return str(image_path)
        try:
            return Path(image_path).resolve().relative_to(self.data_root.resolve()).as_posix()
        except ValueError:
            return str(image_path)

    def resolve_path(self, stored_path: str) -> Path:
        """Inversa de _to_stored_path: ruta local de una imagen guardada en la base."""
        path = Path(stored_path)
        if self.data_root is None or path.is_absolute():
            return path
        candidate = self.data_root / path
        if not candidate.exists() and path.exists():
            # Base indexada antes de guardar rutas relativas a DATA_PATH.
            return path
        return candidate

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial.

        La ruta se guarda relativa a DATA_PATH (ej: dataset/train/<raza>/<imagen>.jpg).
        Para leer la imagen a partir de un registro, usar resolve_path(record.path).
        """
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=self._to_stored_path(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(self.resolve_path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
