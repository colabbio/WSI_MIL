# Scripts de Operación e Ingesta (Runbooks)

Este directorio (`ingestion/`) contiene los scripts operativos ("Import Scripts") diseñados para conectar los datos masivos crudos (imágenes en disco) con el ecosistema estructurado de ColabBio (Bases de datos FHIR, simuladores DICOM).

Esta documentación sirve como guía para entender qué hacen bajo el capó y cómo puedes adaptarlos para **futuros usos** o nuevos datasets.

---

## 1. `fhir-omop-loader.py` (Ingesta de Trazabilidad Clínica)

**Propósito:**  
Extraer las identidades de pacientes de las imágenes físicas y crear registros clínicos sintéticos en el servidor **HAPI FHIR**, estableciendo un enlace duro (`DocumentReference`) entre la ficha clínica y la ruta física de la imagen.

**Cómo funciona bajo el capó:**
1. **Escaneo del Volumen:** Usa `glob` para listar todos los ficheros `.svs` en la ruta montada por Kubernetes (`/mnt/wsi_data/...`).
2. **Extracción (Regex):** Usa una expresión regular para detectar el ID estándar del TCGA (ej. `TCGA-XX-XXXX`).
3. **Generación Sintética:** Genera metadatos *mock* (edad, género) para simular un historial clínico.
4. **API REST:** Envía peticiones HTTP `PUT/POST` al servidor FHIR del clúster (`http://colabbio-hapi-fhir...`).

**Para Futuros Usos (Adaptación):**
Si traes un dataset que no es del TCGA (ej. Hospital Local) y **tienes un CSV real** con los metadatos:
- Elimina la sección de "Generación Sintética".
- Sustituye el bloque `glob` por un `import pandas as pd`, y lee tu CSV.
- Por cada fila del CSV, extrae el ID del paciente, mapea sus columnas (diagnóstico, edad real) al JSON de FHIR, y mantén la lógica del `DocumentReference` para enlazar la imagen.

---

## 2. `convert_to_dicom.sh` (Simulador de PACS / DICOM WSI)

**Propósito:**  
Convertir archivos crudos `.svs` (Aperio) a formato estándar `DICOM WSI` para simular que los datos provienen de un PACS hospitalario real (Orthanc) en lugar de un NAS.

**Cómo funciona bajo el capó:**
El script está diseñado para encapsular la herramienta `wsi2dcm` (proveída típicamente por contenedores Docker de Orthanc).
1. Toma una ruta de entrada (el `.svs` en MedVFS).
2. Genera una carpeta de salida (`DICOM_WSI`).
3. (En producción) Invocaría un comando de conversión que re-empaqueta la pirámide de resoluciones del SVS en frames DICOM.
4. *(Nota de simulación actual)*: Para agilizar las pruebas en el piloto sin consumir horas de CPU, el script actual genera un fichero *dummy* con extensión `.dcm`. Esto es intencional: fuerza a `openslide` a fallar en el dataloader de Python para probar la resistencia a fallos del pipeline al usar los features extraídos.

**Para Futuros Usos (Adaptación):**
Si quieres ingestar datos reales a un PACS Orthanc en el futuro:
- Descomenta la línea del `docker run ... osimis/orthanc-wsi wsi2dcm ...` dentro del script.
- Añade un paso adicional que haga un `curl -X POST` al endpoint `/instances` de tu Orthanc enviando el archivo `.dcm` generado para registrarlo formalmente en la base de datos de imagen médica.
