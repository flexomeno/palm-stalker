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
    parser.add_argument(
        "--dedup-radius-m",
        type=float,
        default=2.0,
        help="Radio en metros para deduplicar detecciones cercanas.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Mantiene todo el pipeline y ademas exporta un archivo de conteo total.",
    )
    parser.add_argument(
        "--output-count",
        type=str,
        default="conteo_total.txt",
        help="Ruta del archivo de salida para el conteo total.",
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
    if args.dedup_radius_m < 0:
        raise ValueError("--dedup-radius-m debe ser >= 0.")

    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError("El TIFF no tiene CRS definido. No se pueden generar coordenadas geograficas.")
        if src.count < 3:
            logger.warning(
                "El raster tiene %s banda(s). DeepForest suele funcionar mejor con RGB (3 bandas).",
                src.count,
            )
        logger.info(
            "Validacion raster OK: size=%sx%s, bands=%s, CRS=%s",
            src.width,
            src.height,
            src.count,
            src.crs,
        )


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
            "score": predictions.get("score", pd.Series([np.nan] * len(predictions))).values,
            "label": predictions.get("label", pd.Series(["palma"] * len(predictions))).values,
            "x_map": np.array(x_map, dtype=float),
            "y_map": np.array(y_map, dtype=float),
        },
        geometry=[Point(x, y) for x, y in zip(x_map, y_map)],
        crs=crs,
    )

    return gdf


def project_to_metric_crs(gdf: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    if gdf.crs is None:
        raise ValueError("El GeoDataFrame no tiene CRS; no es posible proyectar a metros.")

    if gdf.crs.is_projected:
        logger.info("CRS ya proyectado (%s). Se asume unidad metrica.", gdf.crs)
        return gdf.copy()

    estimated_utm = gdf.estimate_utm_crs()
    if estimated_utm is None:
        raise ValueError("No se pudo estimar CRS UTM para calculo de distancias en metros.")

    logger.info("Reproyectando a CRS metrico para analisis de distancia: %s", estimated_utm)
    return gdf.to_crs(estimated_utm)


def deduplicate_by_radius(
    gdf_metric: gpd.GeoDataFrame,
    dedup_radius_m: float,
    logger: logging.Logger,
) -> gpd.GeoDataFrame:
    if gdf_metric.empty or dedup_radius_m <= 0:
        if dedup_radius_m <= 0:
            logger.info("Deduplicacion desactivada (radio <= 0).")
        return gdf_metric

    coords = np.column_stack((gdf_metric.geometry.x.to_numpy(), gdf_metric.geometry.y.to_numpy()))
    tree = cKDTree(coords)
    n = len(coords)
    visited = np.zeros(n, dtype=bool)
    keep_indices = []
    scores = gdf_metric["score"].fillna(0.0).to_numpy(dtype=float)

    for idx in range(n):
        if visited[idx]:
            continue

        neighbor_ids = tree.query_ball_point(coords[idx], r=dedup_radius_m)
        component = []
        stack = list(neighbor_ids)
        while stack:
            current = stack.pop()
            if visited[current]:
                continue
            visited[current] = True
            component.append(current)
            stack.extend(tree.query_ball_point(coords[current], r=dedup_radius_m))

        # Mantiene la deteccion de mayor score por cluster espacial.
        best_idx = max(component, key=lambda i: scores[i])
        keep_indices.append(best_idx)

    dedup_gdf = gdf_metric.iloc[sorted(keep_indices)].copy().reset_index(drop=True)
    dedup_gdf["id"] = np.arange(1, len(dedup_gdf) + 1, dtype=int)
    logger.info(
        "Deduplicacion aplicada con radio %.2fm: %s -> %s detecciones.",
        dedup_radius_m,
        len(gdf_metric),
        len(dedup_gdf),
    )
    return dedup_gdf


def compute_nearest_neighbor_distance(gdf: gpd.GeoDataFrame, logger: logging.Logger) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf["distancia_al_vecino_m"] = []
        return gdf

    coords = np.column_stack((gdf.geometry.x.to_numpy(), gdf.geometry.y.to_numpy()))

    if len(coords) == 1:
        gdf["distancia_al_vecino_m"] = np.nan
        return gdf

    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    gdf["distancia_al_vecino_m"] = distances[:, 1]
    logger.info("Distancias al vecino mas cercano calculadas para %s palmas.", len(gdf))
    return gdf


def export_results(
    gdf_original_crs: gpd.GeoDataFrame,
    output_geojson: str,
    output_csv: str,
    logger: logging.Logger,
) -> None:
    output_geojson_path = Path(output_geojson)
    output_csv_path = Path(output_csv)

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    gdf_original_crs.to_file(output_geojson_path, driver="GeoJSON")
    logger.info("GeoJSON guardado en: %s", output_geojson_path)

    if gdf_original_crs.empty:
        csv_df = pd.DataFrame(columns=["id", "lat", "lon", "distancia_al_vecino_m"])
        csv_df.to_csv(output_csv_path, index=False)
        logger.info("CSV vacio guardado en: %s", output_csv_path)
        return

    gdf_ll = gdf_original_crs.to_crs(epsg=4326)
    csv_df = pd.DataFrame(
        {
            "id": gdf_ll["id"].astype(int),
            "lat": gdf_ll.geometry.y.astype(float),
            "lon": gdf_ll.geometry.x.astype(float),
            "distancia_al_vecino_m": gdf_original_crs["distancia_al_vecino_m"].astype(float),
        }
    )
    csv_df.to_csv(output_csv_path, index=False)
    logger.info("CSV guardado en: %s", output_csv_path)


def export_count(total_count: int, output_count: str, logger: logging.Logger) -> None:
    output_count_path = Path(output_count)
    output_count_path.parent.mkdir(parents=True, exist_ok=True)
    output_count_path.write_text(f"conteo_total,{int(total_count)}\n", encoding="utf-8")
    logger.info("Conteo total guardado en: %s", output_count_path)


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
    gdf_metric = project_to_metric_crs(gdf=gdf, logger=logger)
    gdf_metric = deduplicate_by_radius(
        gdf_metric=gdf_metric,
        dedup_radius_m=args.dedup_radius_m,
        logger=logger,
    )
    gdf_metric = compute_nearest_neighbor_distance(gdf=gdf_metric, logger=logger)

    if gdf_metric.empty:
        gdf_output = gdf_metric
    else:
        gdf_output = gdf_metric.to_crs(gdf.crs)
        gdf_output["distancia_al_vecino_m"] = gdf_metric["distancia_al_vecino_m"].to_numpy()
        gdf_output["x_map"] = gdf_output.geometry.x.to_numpy()
        gdf_output["y_map"] = gdf_output.geometry.y.to_numpy()

    export_results(
        gdf_original_crs=gdf_output,
        output_geojson=args.output_geojson,
        output_csv=args.output_csv,
        logger=logger,
    )

    if args.count_only:
        export_count(total_count=len(gdf_output), output_count=args.output_count, logger=logger)
        logger.info("Modo count-only activo. Conteo total de palmas: %s", len(gdf_output))

    logger.info("Procesamiento finalizado correctamente.")


if __name__ == "__main__":
    main()
