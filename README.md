# Tenant Pilot: IA de Anatomía Patológica (WSI MIL)

Este repositorio contiene el código de investigación del piloto para entrenar un modelo ABMIL (Attention-Based Multiple Instance Learning) utilizando la arquitectura **MedVFS FUSE** de ColabBio.

## 1. Diseño Arquitectónico (MedVFS FUSE-First)

El factor diferencial de este piloto es que **no utiliza conexiones directas a APIs ni PVCs de almacenamiento crudos**.
La IA consume todos los datos a través del **Demonio MedVFS (FUSE)** programado en Go, que actúa como puente POSIX-a-FHIR/OMERO en tiempo real.

### El Archivo Mágico: `wsi-vfs.yaml`
Todo el ecosistema de datos del tenant se define centralizadamente en `datasets/wsi-vfs.yaml`.
Ese archivo le indica a MedVFS cómo conectarse al servidor FHIR, dónde están los SVS locales (o DICOMweb) y dónde está OMERO. Cuando la IA hace `read()`, MedVFS orquesta las llamadas de red por debajo.

> **💡 Mantenimiento de Legacy:** Si ejecutas este código en un clúster de Kubernetes que prohíbe Pods con privilegios FUSE (`securityContext.privileged`), revisa la carpeta `legacy-pure-k8s/` y el archivo `legacy-pure-k8s/README.md` para ver cómo correr el pipeline montando PVCs crudos (el método antiguo).

## 2. Técnica y Estructura

*   `src/`: Código Python del modelo ABMIL.
    *   `AttentionDataset.py`: Construido para leer de `/data/vfs`, confiando en que MedVFS abstrae la localización real de la imagen.
*   `datasets/`: Manifiestos.
    *   `wsi-vfs.yaml`: Configuración central de MedVFS para este tenant.
*   `ingestion/`: Scripts ETL para ingestar datos externos (Runbooks).

## 3. Operación (Runbook)

El ciclo de vida del Tenant asume que primero poblamos las bases de datos maestras y luego dejamos que MedVFS abstraiga la lectura.

### Paso 0: Ingesta Inicial (Poblar el Core)
Antes de que MedVFS pueda operar, necesitamos inyectar los metadatos clínicos (FHIR) y registrar las imágenes (OMERO). Utiliza los scripts ETL:
```bash
# Inyectar pacientes sintéticos a FHIR
kubectl apply -f ingestion/ingestion-job.yaml -n colabbio-dev

# (Opcional) Registrar symlinks en OMERO 
kubectl apply -f legacy-pure-k8s/omero-import-job.yaml -n colabbio-dev
```

### Paso A: Materialización de Metadatos (Setup)
MedVFS extrae la estructura tabular desde FHIR y los thumbnails desde OMERO.
```bash
medvfs setup --config datasets/wsi-vfs.yaml
```

### Paso B: Arrancar Entrenamiento (Streaming FUSE)
Ejecuta el contenedor de IA inyectando MedVFS como sidecar. La IA leerá la montura local `/data/vfs` a toda velocidad.
```bash
# Ejemplo usando Docker local:
docker run -v $(pwd)/datasets/wsi-vfs.yaml:/etc/medvfs/vfs.yaml \
           --privileged \
           colabbio/medvfs:latest serve --config /etc/medvfs/vfs.yaml
```
