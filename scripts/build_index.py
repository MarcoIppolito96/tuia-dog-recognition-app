"""Construye la base vectorial indexando el dataset con extract_embedding (Etapa 1).

Requiere haber implementado SimilarityService.extract_embedding (o, si
EMBEDDING_MODEL != baseline, ClassifierService.extract_custom_embedding).
Si USE_PGVECTOR=true, la base de datos debe estar corriendo
(`docker compose up postgres -d`).

Uso:
    python scripts/build_index.py [--split train] [--limit 0]

Dentro de Docker (con `docker compose up -d` corriendo):
    docker compose exec backend python /scripts/build_index.py [--split train]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if not (SRC / "lib").is_dir():
    # Dentro del contenedor del backend: src/ esta montado en /app (WORKDIR) y scripts/ en /scripts.
    ROOT = SRC = Path.cwd()
sys.path.insert(0, str(SRC))

# Usa la misma configuracion que el backend local (src/.env, con paths relativos a src/).
# Si src/.env no existe, se usan los defaults con paths relativos a la raiz del repo.
os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", default="train", help="Subcarpeta del dataset a indexar (train/valid/test)."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Maximo de imagenes por raza (0 = todas)."
    )
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_similarity, build_store
    from lib.config import settings

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    if settings.embedding_model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(settings.embedding_model)
        extractor = classifier.extract_custom_embedding
    similarity.extract_embedding = extractor  # type: ignore[method-assign]

    # Splits redefinidos en 1-pre-processing.ipynb (sin duplicados ni fuga entre splits).
    # Si el archivo existe se indexa el split pedido desde ahí; si no, las carpetas originales.
    splits_csv = settings.data_path / "splits" / "splits.csv"
    if splits_csv.is_file():
        with open(splits_csv, encoding="utf-8") as f:
            filas = [fila for fila in csv.DictReader(f) if fila["split"] == args.split]
        por_raza: dict[str, int] = {}
        for fila in filas:
            raza = fila["raza"]
            if args.limit and por_raza.get(raza, 0) >= args.limit:
                continue
            similarity.index_image(
                str(settings.dataset_path / fila["ruta"]),
                raza,
                {"split": args.split, "model": settings.embedding_model},
            )
            por_raza[raza] = por_raza.get(raza, 0) + 1
        for raza in sorted(por_raza):
            print(f"{raza}: {por_raza[raza]} imagenes indexadas")
        print(
            f"Total: {sum(por_raza.values())} embeddings almacenados "
            f"(modelo: {settings.embedding_model}, splits: {splits_csv})"
        )
        return

    root = settings.dataset_path / args.split
    if not root.is_dir():
        sys.exit(f"No existe {root}. Descarga el dataset con scripts/download_dataset.py")

    total = 0
    for breed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if args.limit and count >= args.limit:
                break
            similarity.index_image(
                str(image_path),
                breed_dir.name,
                {"split": args.split, "model": settings.embedding_model},
            )
            count += 1
            total += 1
        print(f"{breed_dir.name}: {count} imagenes indexadas")

    print(f"Total: {total} embeddings almacenados (modelo: {settings.embedding_model})")


if __name__ == "__main__":
    main()
