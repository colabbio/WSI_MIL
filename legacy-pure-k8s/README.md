# Tenant Project: WSI_MIL (ABMIL - TCGA)

Este repositorio es el proyecto oficial piloto (Tenant) para la plataforma **ColabBio**. Adapta un modelo monolítico de Deep Learning (Attention-Based MIL) sobre imágenes histológicas (WSI) masivas, transformándolo en un microservicio nativo de la nube.

Esta documentación está dividida en tres perspectivas clave: **Diseño, Técnica y Operación**.

---

## 1. Documentación de Diseño (Arquitectura)

El objetivo de diseño es **cero acoplamiento**: el código científico que entrena el modelo de IA no debe saber nada sobre Kubernetes, FHIR, o la infraestructura de red.

- **Abstracción de Datos:** El código Python (`src/`) lee las imágenes WSI directamente desde `/mnt/wsi_data`.
- **Inmutabilidad del Código Base:** No se han añadido *if/else* en el código para lidiar con formatos (SVS vs DICOM). Toda esa complejidad se ha empujado a la infraestructura (Patrón de Configuración Dual MedVFS).
- **Trazabilidad FHIR:** Los datos clínicos estructurados y las URL físicas de los ficheros de imagen se inyectan en HAPI FHIR para que los visores web de ColabBio puedan renderizarlos sin tocar este repositorio de IA.

---

## 2. Documentación Técnica (Componentes)

La infraestructura como código (IaC) de este tenant está modularizada:

### Código Fuente (Intacto)
- `src/AttentionDataset.py`: Dataloader. Preparado para usar `openslide` o asimilar fallos (útil para el fallback de HDF5 al apuntar a DICOM).
- `src/training.py`: Pipeline principal de entrenamiento de Pytorch.

### Infraestructura Dual (MedVFS)
Tenemos dos modos de montar los datos al contenedor, definidos en `datasets/`:
- **Filesystem Mode:** `medvfs-local.yaml` (apunta directamente al NAS `/data/home/lolmo/histology`).
- **DICOM Mode:** `medvfs-dicom.yaml` (simula un acceso a PACS o DICOM-web).

### Kubernetes Workflows
Definidos en `pipelines/`:
- `training-job-local.yaml` / `training-job-dicom.yaml`: Lanza el worker de IA, inyecta la CPU/Memoria necesaria, y le monta el PVC de MedVFS apropiado.
- `ingestion-job.yaml`: Lanza un contenedor efímero que ejecuta el script de ingesta de FHIR.

---

## 3. Documentación de Operación (Runbook)

Si eres el Operador/Administrador a cargo de correr este Tenant Project en el clúster `colabbio-dev`, sigue este Runbook:

### Paso A: Inicializar los Datos (Opcional, si el clúster es nuevo)
Para simular el entorno PACS, primero convierte una muestra `.svs` a DICOM WSI.
```bash
chmod +x ingestion/convert_to_dicom.sh
./ingestion/convert_to_dicom.sh
```

### Paso B: Ingesta de Trazabilidad en FHIR
Debemos poblar la base de datos central de ColabBio con los metadatos clínicos de estos pacientes, de modo que el Hospital/Plataforma Web sepa que existen y dónde están sus imágenes WSI.
```bash
# 1. Crear el PVC local para que el script pueda leer las IDs reales
kubectl apply -f datasets/medvfs-local.yaml -n colabbio-dev

# 2. Lanzar el job de ingesta sintética
kubectl apply -f pipelines/ingestion-job.yaml -n colabbio-dev

# 3. Esperar y verificar
kubectl wait --for=condition=complete job/tcga-synthetic-ingestion -n colabbio-dev --timeout=60s
```

### Paso C: Enlazar Imágenes en OMERO (In-Place Import)
Para que el visualizador médico pueda mostrar las imágenes que tu IA está entrenando sin duplicar datos en disco, lanza el script de OMERO que inyecta symlinks aislados por Tenant.
```bash
kubectl apply -f pipelines/omero-import-job.yaml -n colabbio-dev
```

### Paso D: Lanzar el Entrenamiento de IA
Dependiendo de la política de datos que quieras usar, lanza el pipeline en uno de los dos modos:

**Opción 1: Entrenar leyendo de SVS en disco (Filesystem)**
```bash
kubectl apply -f datasets/medvfs-local.yaml -n colabbio-dev
kubectl apply -f pipelines/training-job-local.yaml -n colabbio-dev
```

**Opción 2: Entrenar simulando Backend DICOM**
```bash
kubectl apply -f datasets/medvfs-dicom.yaml -n colabbio-dev
kubectl apply -f pipelines/training-job-dicom.yaml -n colabbio-dev
```

Monitoriza el progreso de la IA usando:
```bash
kubectl logs -f job/abmil-training-job-local -n colabbio-dev
```

---
> **💡 ¿Quieres entender los scripts o adaptarlos a otros datos?**  
> Revisa la [Documentación de Scripts de Ingesta (Runbooks)](ingestion/README.md) para detalles sobre cómo funciona FHIR y la conversión DICOM bajo el capó.
