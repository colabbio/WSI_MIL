#!/bin/bash
set -euo pipefail
# ==============================================================================
# omero-importer.sh — Legacy Demo Seed Script (Superseded by Proxy/MedVFS on Edge)
# ==============================================================================
# NOTA ARQUITECTÓNICA (CB-SPEC-PROXY-VNA-018):
# En despliegues hospitalarios reales (p. ej. Red Española de Cáncer / SNS / CIBERONC)
# y en HPC (MareNostrum 5), NO se ejecuta 'omero import':
#   1. Si el hospital tiene VNA/IMS/OMERO propio -> 'proxy.yaml' (mode: proxy / hybrid-staging)
#      conecta en Zero-Copy o pre-cachea solo la cohorte aprobada por el DPO en NVMe con TTL.
#   2. Si las WSIs están en NFS/GPFS -> 'MedVFS' (vfs.yaml) las monta y sirve directamente.
# Este script SOLO se activa cuando OMERO_PROXY_MODE="local" (entornos Demo Standalone).
# ==============================================================================

OMERO_PROXY_MODE="${OMERO_PROXY_MODE:-proxy}"

if [[ "${OMERO_PROXY_MODE}" != "local" ]]; then
    echo "================================================================================"
    echo "[OMERO-IMPORTER] SKIPPED: OMERO_PROXY_MODE='${OMERO_PROXY_MODE}' (Zero-Copy Proxy / MedVFS activo)."
    echo "En nodos hospitalarios (Red Española de Cáncer) el acceso a WSI se realiza vía"
    echo "proxy.yaml (Nginx OMERO Gateway / Orthanc DICOMweb Bridge / MedVFS) sin importar"
    echo "ni duplicar metadatos en un OMERO.server local."
    echo "================================================================================"
    exit 0
fi

OMERO_HOST="${OMERO_HOST:-colabbio-omero-server}"
OMERO_PORT="${OMERO_PORT:-4064}"
OMERO_USER="${OMERO_USER:-root}"
OMERO_PASSWORD="${OMERO_PASSWORD:-omero}"
WSI_MOUNT_DIR="${WSI_MOUNT_DIR:-/mnt/wsi_data/BRCA_AWS/BRCA_raw}"
TENANT_GROUP="${TENANT_GROUP:-WSI_MIL_TCGA}"
MAX_IMPORT_BATCH="${MAX_IMPORT_BATCH:-50}"

if ! command -v omero >/dev/null 2>&1; then
    echo "[OMERO-IMPORTER] 'omero' CLI no instalado en este contenedor; omitiendo seed local."
    exit 0
fi

omero login -s "${OMERO_HOST}" -p "${OMERO_PORT}" -u "${OMERO_USER}" -w "${OMERO_PASSWORD}"
omero group add "${TENANT_GROUP}" --perms='rwra--' 2>/dev/null || true

shopt -s nullglob
SLIDE_FILES=("${WSI_MOUNT_DIR}"/*.{svs,ndpi,tif,tiff})
COUNT=0
for FILE in "${SLIDE_FILES[@]}"; do
    (( COUNT >= MAX_IMPORT_BATCH )) && break
    omero import --transfer=ln_s --group="${TENANT_GROUP}" --skip=all "${FILE}"
    COUNT=$((COUNT + 1))
done
echo "[OMERO-IMPORTER] Modo Demo Local: importadas ${COUNT} WSI(s) en ${TENANT_GROUP}."
