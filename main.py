import argparse
import logging
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from deepforest import main as deepforest_main
from scipy.spatial import cKDTree
from shapely.geometry import Point


def configure_logger() -> logging.Logger:
    logger = logging.getLogger("palmas_cera")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecta palmas de cera en ortofoto, calcula coordenadas y distancia al vecino."
    )
    parser.add_argument(
        "--input-tif",
        type=str,
        required=True,
        help="Ruta al ortomosaico .tif (montado en el contenedor).",
    )
    parser.add_argument(
        "--output-geojson",
        type=str,
        default="resultados_palmas.geojson",
        help="Ruta de salida para GeoJSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="reporte.csv",
        help="Ruta de salida para CSV.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=800,
        help="Tamano del parche para predict_tile.",
    )
    parser.add_argument(
        "--patch-overlap",
        type=float,
        default=0.15,
        help="Traslape entre parches [0, 1).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Umbral de confianza para conservar detecciones.",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace, logger: logging.Logger) -> None:
    tif_path = Path(args.input_tif)
    if not tif_path.exists():
        raise FileNotFoundError(f"No se encontro el archivo TIFF: {tif_path}")
    if tif_path.suffix.lower() not in [".tif", ".tiff"]:
        raise ValueError("El archivo de entrada debe ser .tif o .tiff")

    if not (0 <= args.patch_overlap < 1):
        raise ValueError("--patch-overlap debe estar en [0, 1).")
    if not (0 <= args.score_threshold <= 1):
        raise ValueError("--score-threshold debe estar en [0, 1].")
    if args.patch_size < 200:
        logger.warning("patch-size muy pequeno puede degradar precision y rendimiento.")


def initialize_model(logger: logging.Logger):
    logger.info("Inicializando modelo DeepForest (CPU, release pre-entrenado)...")
    model = deepforest_main.deepforest()
    model.use_release()
    model.config["gpus"] = 0
    return model


def run_detection(
    model,
    tif_path: str,
    patch_size: int,
    patch_overlap: float,
    score_threshold: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info(
        "Iniciando deteccion por tiles: patch_size=%s, overlap=%.2f, threshold=%.2f",
        patch_size,
        patch_overlap,
        score_threshold,
    )

    predictions = model.predict_tile(
        raster_path=tif_path,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        iou_threshold=0.15,
        return_plot=False,
    )

    if predictions is None or predictions.empty:
        logger.warning("No se detectaron objetos en la imagen.")
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])

    if "score" in predictions.columns:
        before = len(predictions)
        predictions = predictions[predictions["score"] >= score_threshold].copy()
        logger.info(
            "Detecciones filtradas por score: %s -> %s",
            before,
            len(predictions),
        )

    return predictions


def pixel_to_map_coordinates(
    predictions: pd.DataFrame, tif_path: str, logger: logging.Logger
) -> gpd.GeoDataFrame:
    with rasterio.open(tif_path) as src:
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height
        logger.info(
            "Raster abierto. size=%sx%s, CRS=%s",
            width,
            height,
            crs,
        )

    if predictions.empty:
        gdf_empty = gpd.GeoDataFrame(
            {
                "id": [],
                "score": [],
                "label": [],
                "x_map": [],
                "y_map": [],
            },
            geometry=[],
            crs=crs,
        )
        return gdf_empty

    centroids_x = ((predictions["xmin"] + predictions["xmax"]) / 2.0).to_numpy()
    centroids_y = ((predictions["ymin"] + predictions["ymax"]) / 2.0).to_numpy()

    x_map, y_map = rasterio.transform.xy(
        transform,
        centroids_y,
        centroids_x,
        offset="center",
    )

    gdf = gpd.GeoDataFrame(
        {
            "id": np.arange(1, len(predictions) + 1, dtype=int),
            "score": predictions.get("score", pd.Series([None] * len(predictions))).values,
            "label": predictions.get("label", pd.Series(["palma"] * len(predictions))).values,
            "x_map": np.array(x_map, dtype=float),
            "y_map": np.array(y_map, dtype=float),
        },
        geometry=[Point(x, y) for x, y in zip(x_map, y_map)],
        crs=crs,
    )

    return gdf


def compute_nearest_neighbor_distance(gdf: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["distancia_al_vecino"] = []
        return gdf

    coords = np.column_stack((gdf["x_map"].to_numpy(), gdf["y_map"].to_numpy()))

    if len(coords) == 1:
        gdf["distancia_al_vecino"] = np.nan
        return gdf

    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    gdf["distancia_al_vecino"] = distances[:, 1]
    logger.info("Distancias a vecino mas cercano calculadas para %s palmas.", len(gdf))
    return gdf


def export_results(
    gdf: gpd.GeoDataFrame, output_geojson: str, output_csv: str, logger: logging.Logger
) -> None:
    output_geojson_path = Path(output_geojson)
    output_csv_path = Path(output_csv)

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(output_geojson_path, driver="GeoJSON")
    logger.info("GeoJSON guardado en: %s", output_geojson_path)

    if gdf.empty:
        csv_df = pd.DataFrame(columns=["id", "lat", "lon", "distancia_al_vecino"])
        csv_df.to_csv(output_csv_path, index=False)
        logger.info("CSV vacio guardado en: %s", output_csv_path)
        return

    gdf_ll = gdf.to_crs(epsg=4326)
    csv_df = pd.DataFrame(
        {
            "id": gdf_ll["id"].astype(int),
            "lat": gdf_ll.geometry.y.astype(float),
            "lon": gdf_ll.geometry.x.astype(float),
            "distancia_al_vecino": gdf["distancia_al_vecino"].astype(float),
        }
    )
    csv_df.to_csv(output_csv_path, index=False)
    logger.info("CSV guardado en: %s", output_csv_path)


def main():
    logger = configure_logger()
    args = parse_args()
    validate_inputs(args, logger)

    tif_path = os.path.abspath(args.input_tif)
    logger.info("Entrada TIFF: %s", tif_path)

    model = initialize_model(logger)
    predictions = run_detection(
        model=model,
        tif_path=tif_path,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        score_threshold=args.score_threshold,
        logger=logger,
    )
    logger.info("Total detecciones finales: %s", len(predictions))

    gdf = pixel_to_map_coordinates(predictions=predictions, tif_path=tif_path, logger=logger)
    gdf = compute_nearest_neighbor_distance(gdf=gdf, logger=logger)

    export_results(
        gdf=gdf,
        output_geojson=args.output_geojson,
        output_csv=args.output_csv,
        logger=logger,
    )

    logger.info("Procesamiento finalizado correctamente.")


if __name__ == "__main__":
    main()
