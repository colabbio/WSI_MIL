# Módulo de Transcriptómica Espacial & Vision Transformers (UCSF / CNAG)

Este subpaquete contiene la extensión multimodal de **ColabBio** para procesar **Transcriptómica Espacial (10x Visium, Xenium, Stereo-seq)** combinada con imágenes histológicas de alta resolución (H&E).

---

## 🎯 Hoja de Ruta para Lorenzo (PhD Scope)

### 1. ¿Qué está ya resuelto por la plataforma? (Infraestructura ColabBio)
- **Virtualización FUSE (`MedVFS`):** Lectura directa de matrices `.h5` / AnnData y ficheros de coordenadas `tissue_positions.csv` desde almacenamiento sin descargas a disco.
- **Cargador Unificado de ViTs (`FoundationViTLoader.py`):** Carga transparente de *UNI2-h*, *Virchow 2*, *Prov-GigaPath* y *DINOv2 Large* con token o sin token.
- **Orquestación en la nube (`Nextflow DSL2`):** Flag `enable_spatial_transcriptomics` en el pipeline de Nextflow con registro en MLflow.
- **Inferencia y Servicio:** Exportación a ONNX y hot-reload en Triton Inference Server para el visor web (`cb_path_viewer`).

---

### 2. ¿Qué le toca a Lorenzo en `abMIL`? (Tareas Pendientes en WSI)
- **Ordinal ISUP Loss / Soft-Margin:** En `MultiTaskMIL`, el cabezal de ISUP predice con `CrossEntropyLoss` categórica pura. Un ISUP 1 clasificado como ISUP 2 penaliza igual que clasificado como ISUP 5. Implementar una **pérdida ordinal (Quadratic Weighted Kappa Loss o Earth Mover's Distance)** mejora la sensibilidad diagnóstica.
- **Atención Multiescala con diferentes magnificaciones:** `abMIL` actualmente recibe parches a 20x. Extender la compuerta de atención para recibir parches emparejados a 20x (detalles celulares) y 5x (arquitectura tisular macro).
- **Benchmarking formal para el Paper:** Ejecutar el `study.tcga-prad.yaml` y comparar cuantitativamente las métricas de:
  `MultiTaskMIL` vs. `abMIL (single task)` vs. `TransMIL` vs. `CoxMIL`.

---

### 3. ¿Qué le toca a Lorenzo en `spatial`? (El Core Científico del Doctorado)

#### Tarea A: `align_spots_to_tiles.py` (Preparación Biológica)
- Archivo plantilla: `scripts/align_spots_to_tiles.py`.
- **Qué tiene que decidir y afinar:**
  1. El filtrado de genes en `scanpy`: definir el umbral de dispersión para seleccionar los 250 genes marcadores más variables o firmas oncológicas de próstata (*AR*, *KLK3*, *TMPRSS2*, *ERG*, *PTEN*, *FOXA1*).
  2. El tamaño del tile alrededor del spot: en Visium el spot mide $55\,\mu\text{m}$, probar qué recorte morfológico maximiza la correlación ($112\times112$ vs. $224\times224$ píxeles).

#### Tarea B: Implementar `SpatialViTDualAttention.py` (Arquitectura Multimodal)
- **Propósito:** Predecir los perfiles de expresión de los 250 genes directamente desde las teselas H&E.
- **Componentes a desarrollar:**
  1. **Self-Attention cruzada entre spots vecinos:** Construir la matriz de adyacencia de grafos o auto-atención espacial para modelar la difusión de citocinas y la interacción tumor-estroma.
  2. **Función de Pérdida Biológica:** Combinar el error cuadrático medio con la Correlación de Pearson por gen:
     $$\mathcal{L} = \text{MSE}(Y, \hat{Y}) + \lambda (1 - \text{Pearson}(Y, \hat{Y}))$$
  3. **Validación en cohortes de UCSF y CNAG:** Evaluar qué genes son altamente predecibles desde la imagen (ej. genes estromales como *COL1A1* suelen dar $r > 0.7$, mientras que mutaciones puntuales dan correlación más baja).
