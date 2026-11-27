# WSI_MIL — Interpretable Multiple Instance Learning & Polarized Attentional Certainty (PAC)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

PyTorch implementation of Attention-Based Multiple Instance Learning (**ABMIL**, **TransMIL**, **ProtoMIL**, **CoxMIL**, **MultiTaskMIL**) and **Polarized Attentional Certainty (PAC)** for computational pathology Whole Slide Images (WSIs), designed to consume streaming datasets through **MedVFS** (`cb-vfs`) FUSE mounts.

## Architecture (MedVFS FUSE-First)

Rather than coupling training code to raw hospital APIs or duplicating terabytes of WSI files onto local PVCs, `WSI_MIL` reads slides and manifests from `/data/vfs` exposed by **MedVFS**:

- `src/models/`: MIL architectures (`AttentionModel.py`, `MIL_model.py`, `TransMIL.py`, `ProtoMIL.py`, `CoxMIL.py`, `MultiTaskMIL.py`).
- `datasets/wsi-vfs.yaml`: Declarative MedVFS manifest binding FHIR/OMOP clinical metadata, DICOMweb/OMERO imaging, and local/cloud WSI stores.
- `ingestion/`: ETL helper scripts for populating synthetic/test FHIR and OMERO instances.
- `legacy-pure-k8s/`: Legacy PVC-based Kubernetes manifests for clusters that restrict privileged FUSE pods.

## Documentation & Thesis

- **[Polarized Attentional Certainty (PAC) Preprint](https://github.com/colabbio/SlideLab/blob/v2.0/docs/pac_article.md)** — $[-1, 1]$ interpretability metric combining attention magnitude and class-logit certainty across colorectal cancer MSI-H/MSS WSIs and B-ALL flow cytometry.
- **[SlideLab Preprint](https://github.com/colabbio/SlideLab/blob/v2.0/docs/slidelab_article.md)** — Modular WSI preprocessing, 1,000-WSI TCGA-COAD benchmarks, and zero-disk MedVFS streaming.
- **[UC San Diego M.S. Thesis Defense Slides (PDF)](https://github.com/colabbio/SlideLab/blob/v2.0/docs/thesis_presentation.pdf)** · **[Open in Google Slides](https://docs.google.com/presentation/d/1AZUdJUrf6N98_nFGE76qFn8rHJB9XNI0SPVWnQrF70k/present)** — Lorenzo Olmo Marchal (2025).

## Quick Start

```bash
# 1. Materialize clinical metadata via MedVFS
medvfs setup --config datasets/wsi-vfs.yaml

# 2. Mount WSI data via MedVFS FUSE and run training
docker run -v $(pwd)/datasets/wsi-vfs.yaml:/etc/medvfs/vfs.yaml \
           --privileged \
           ghcr.io/colabbio/medvfs:latest serve --config /etc/medvfs/vfs.yaml
```

## License

Licensed under the [Apache License 2.0](LICENSE) — Copyright 2026 ColabBio.
