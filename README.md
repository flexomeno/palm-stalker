# Palm Stalker - Deteccion de Palma de Cera en Ortofoto

Proyecto para detectar palmas de cera sobre una ortofoto grande (`.tif`), contar individuos, obtener coordenadas geograficas y calcular distancia al vecino mas cercano.

El flujo esta pensado para archivos pesados (ej. 11GB) usando procesamiento por tiles para reducir consumo de memoria.

## Estructura del proyecto

- `Dockerfile`: imagen de ejecucion con dependencias de GDAL/PROJ y Python.
- `requirements.txt`: librerias Python del pipeline.
- `main.py`: script principal de deteccion, georreferenciacion y exportacion.

## Tecnologias usadas

- `deepforest` para deteccion de objetos (modelo preentrenado con `use_release()`).
- `rasterio` para manejo de raster geoespacial.
- `geopandas` y `shapely` para datos geograficos vectoriales.
- `scipy` para calculo de vecino mas cercano.

## Requisitos

- Docker instalado y funcionando.
- Ortofoto `.tif` disponible en una carpeta local que se montara dentro del contenedor.
- Equipo objetivo: Mac Intel con 32GB RAM (procesamiento en CPU).

## Construccion de la imagen

Desde la carpeta del proyecto:

```bash
docker build -t palm-stalker:cpu .
```

## Ejecucion

Ejemplo usando el TIFF incluido en esta carpeta:

```bash
docker run --rm \
  -v "/Users/flexomeno/Documents/Proyectos/Palm-stalker:/data" \
  palm-stalker:cpu \
  --input-tif "/data/Ortofotomosaico 10 cm-px.tif" \
  --output-geojson "/data/resultados_palmas.geojson" \
  --output-csv "/data/reporte.csv" \
  --patch-size 800 \
  --patch-overlap 0.15 \
  --score-threshold 0.30 \
  --dedup-radius-m 2.0
```

## Parametros del script (`main.py`)

- `--input-tif` (requerido): ruta al archivo `.tif`.
- `--output-geojson` (opcional): salida GeoJSON. Default: `resultados_palmas.geojson`.
- `--output-csv` (opcional): salida CSV. Default: `reporte.csv`.
- `--patch-size` (opcional): tamano de tile para inferencia. Default: `800`.
- `--patch-overlap` (opcional): traslape de tiles en `[0,1)`. Default: `0.15`.
- `--score-threshold` (opcional): confianza minima de deteccion en `[0,1]`. Default: `0.30`.
- `--dedup-radius-m` (opcional): radio en metros para deduplicar detecciones cercanas. Default: `2.0`.

## Salidas generadas

1. `resultados_palmas.geojson`
   - Geometria puntual por palma detectada.
   - Incluye campos de identificador, score, etiqueta y distancia al vecino.
2. `reporte.csv`
   - Columnas: `id`, `lat`, `lon`, `distancia_al_vecino_m`.

## Flujo de procesamiento

1. Carga del modelo DeepForest preentrenado.
2. Inferencia por tiles (`predict_tile`) para evitar cargar todo el raster en memoria.
3. Conversion de centroides de pixeles a coordenadas reales usando CRS y transform del raster.
4. Reproyeccion automatica a CRS metrico (UTM estimado) para distancias en metros.
5. Deduplicacion espacial por radio para reducir cajas repetidas en bordes de tile.
6. Calculo de distancia al vecino mas cercano con `cKDTree`.
7. Exportacion de resultados a GeoJSON y CSV.

## Monitoreo y rendimiento

- El script usa logs con nivel `INFO` para seguir el avance.
- En archivos muy grandes, el tiempo de ejecucion puede ser alto en CPU.
- Si hay pocas detecciones, puede probar:
  - aumentar `patch-size` (si hay RAM disponible),
  - reducir `score-threshold`,
  - ajustar `patch-overlap` para capturar objetos en bordes de tile.

## Nota de calidad de deteccion

El modelo preentrenado de DeepForest esta orientado a copas de arboles en general. Para mejorar precision especifica sobre palma de cera, se recomienda una etapa posterior de ajuste fino (fine-tuning) con muestras etiquetadas de la zona objetivo.
