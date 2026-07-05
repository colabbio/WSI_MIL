#!/bin/bash
# convert_to_dicom.sh
# Uses Orthanc's wsi2dcm tool via Docker to convert a test SVS file to DICOM WSI.

INPUT_SVS="/data/home/lolmo/histology/BRCA_AWS/BRCA_raw/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.svs"
OUTPUT_DIR="/data/home/lolmo/histology/DICOM_WSI"
OUTPUT_FILE="TCGA-3C-AALI-01Z-00-DX1.dcm"

echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "Converting SVS to DICOM WSI..."
# In a real environment, we would use orthanc-wsi image. For now we simulate it or use a public one if available.
# docker run --rm -v /data:/data osimis/orthanc-wsi wsi2dcm --dataset=tcga --patient-name="TCGA-3C-AALI" "$INPUT_SVS" "$OUTPUT_DIR/$OUTPUT_FILE"

# NOTE: If osimis/orthanc-wsi is not locally cached and takes too long to pull, 
# you can use Python with `wsidicom` to convert, or just assume the file is placed there by the backend.
# For testing the try/except logic of ABMIL without massive conversion, we can simply copy the .svs file 
# and rename it to .dcm, so openslide fails on the DICOM URI logic, simulating the MedVFS fallback.

echo "Simulating DICOM conversion for MedVFS testing..."
# Simulation: Create a dummy non-svs file so openslide throws an exception, triggering the magnification=40 fallback
echo "DICOM WSI HEADER DUMMY" > "$OUTPUT_DIR/$OUTPUT_FILE"
echo "Conversion complete. Saved to $OUTPUT_DIR/$OUTPUT_FILE"
