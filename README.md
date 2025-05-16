# BrainStem X – Intensity-Clustering Pipeline for Brainstem MRI

BrainStem X is an end‑to‑end pipeline for detecting and quantifying signal abnormalities throughout any homogenous brain region, in particular the brainstem, with focus on correlating T2‑FLAIR hyper‑intensity and T1 hypo‑intensity clusters and enabling correlation with other modalities like DWI/SWI. The analysis treats the brainstem or other segmented region as a whole structural unit, using the Harvard-Oxford atlas for segmentation.

*Status*: this is currently a very unpolished and early iteration, under active development as of May 2025. For a more comprehensive and mature implementation, refer to https://github.com/myztery-neuroimg/brainstemx-full

## Technical Features

This pipeline implements a principled approach to brainstem intensity analysis through signal normalization and statistical characterisation, integrating classical image processing techniques with automated report generation, web-UI and multimodal clustering analysis – all implemented in Python with computation-intensive operations delegated to optimized ANTs and FSL libraries. Synthetic data generation also helps in validation and potentially unrelated training data generation.

A key technical approach is the unsupervised clustering approach - the idea is that a pure first-principals approach to analysis eliminates the need for labeled training data, thus avoiding the biases inherent in supervised deep learning methods on this type of data.

* **Advanced Preprocessing** of NiFTI standard format neuroimages – Implements ANTs N4 bias field correction with modality-specific parameters (separate configurations for FLAIR and other sequences), followed by noise reduction and intensity normalization, ensuring consistent signal characteristics even with diverse acquisition protocols.

* **Skull-stripping** – Robust, progressive skull-stripping strategy with ANTs as primary method and SynthStrip fallback for difficult cases, advancing brain extraction across acquisition protocols.

* **Multimodal Quantitative Analysis** – coregistration of T1, FLAIR, SWI, and DWI sequences with voxel-wise statistical correlation between modality-specific signal characteristics. The overlap analysis quantifies T1 hypointensity, SWI susceptibility effects, and DWI signal alterations, providing multidimensional characterization of detected abnormalities that exceeds standard intensity-only approaches.

* **Unsupervised Adaptive Thresholding** with statistical modeling of normal-appearing white matter to establish sequence-specific, patient-adaptive intensity thresholds. This method avoids the limitations of fixed thresholding while eliminating biases introduced by supervised deep learning approaches.

* **Comprehensive Quality Assurance** with metadata validation, geometric verification, and >20 quantitative QA metrics with interactive visualization through a Dash-based 3D browser. The pipeline automatically aborts on critical integrity failures, preventing propagation of input errors.

* **AI-Enhanced Radiological Reporting** – Integrates quantitative cluster metrics and multiple visualization planes with GPT-4.1-vision to generate structured radiological reports in standardized DOCX format. This implementation bridges the gap between computational analysis and research interpretation.

* **Parametric Synthetic Data Generation** via `generate_synthetic_data.py` for creating realistic brainstem lesion patterns with configurable spatial and intensity properties.

## Quick install

```
git clone https://github.com/myztery-neuroimg/brainstemx.git
brew install ants fsl c3d dcm2niix # or apt‑get on Linux, accept licence terms as needed
export ANTSPATH="/usr/local/opt/ants/bin" #or update as appropriate
export FSLDIR="/usr/local/opt/fsl" #or update as appropriate
export PATH="$ANTSPATH:$FSLDIR/bin:$PATH" #or configure via ~/.profile or similar
cd brainstemx
uv init; uv venv; uv sync #python -m venv .venv #if not using uv
source .venv/bin/activate
uv add -r requirements.txt / uv pip install -r requirements.txt #or pip install -r requirements.txt
```

## Typical run

```
uv run python -m brainstemx.cli\
 --flair data/sub‑001_FLAIR.nii.gz\
  --t1   data/sub‑001_T1w.nii.gz\
  --out  results/ID01

uv run python -m brainstemx.web_visualiser.py --root results/ID01

uv run python -m brainstemx.report_generator\
  --subj results/ID01\
  --key  "$OPENAI_API_KEY"
```

## Components 

- `validate_inputs.py` Performs comprehensive quality checks on input data, including verification of acquisition parameters and voxel geometry. The pipeline terminates with specific error codes when critical issues are detected, preventing downstream analysis failures.
- `pipeline.py` Executes the full processing workflow including critical N4 bias field correction, intensity normalization, and registration steps. The pipeline generates lesion masks at multiple statistical thresholds (1.5-3.0 SD) and produces comprehensive outputs including `outputs.json` and `analysis.csv` with detailed morphometric and intensity metrics for each detected cluster.
- `qa.py` Implements hierarchical quality assessment, returning standardized status codes (0=complete, 1=invalid, 2=incomplete) and generating `qa_report.json` with comprehensive quality metrics.
- `report_generator.py` Integrates quantitative cluster metrics with visual representations for GPT-4.1-vision analysis, producing structured radiological reports in DOCX format with findings and impressions sections, suitable for research purposes but not appropriate for clinical interpretation without human radiologist feedback as this is merely a potential supplement to expert interpretation, not a replacement.
- Multimodal analysis automatically detects and incorporates available DWI and SWI sequences, calculating statistical overlap between intensity abnormalities across modalities to provide more specific lesion characterization than single-modality approaches.

## Limitations and Requirements

- **Input Requirements**: The pipeline requires 1mm isotropic T1 images for optimal performance, as the brainstem segmentation relies on the Harvard-Oxford atlas in 1mm MNI space.
- **Atlas Support**: Currently only the Harvard-Oxford atlas is supported for brainstem segmentation. This limitation has benefits, though, it makes the pipeline quite adaptable to other (homogenous) atlas regions with minimal reconfiguration
- **ANTs Parameters**: Several ANTs registration and processing parameters are hardcoded, this will be fixed in upcoming changes.

## Repository layout

```
src/
 ├── core.py                  # utilities, registration, clustering
 ├── validate_inputs.py       # JSON+NIfTI pre‑flight guard
 ├── pipeline.py              # end‑to‑end processing
 ├── qa.py                    # post‑pipeline QA / validation
 ├── postprocess.py           # cluster metrics & csv (see docs)
 ├── report_generator.py      # GPT‑4.1‑vision → DOCX
 ├── generate_synthetic_data.py #Geneation of synthetic data with hyperintense FLAIR cluster(s) of different intensitied and geomorphical attributes along with corresponding hypointense T1 clustering
 ├── web_visualiser.py        # Dash 3‑plane browser
 └── cli.py                   # batch / single‑subject wrapper
requirements.txt
```

## Licence

MIT licence for BrainStem X; external toolkits keep their own licences.

## Collaboration 

PRs are welcomed as is feedback on the pipeline methodology or other neuroradiological aspects.

For pull requests, methodological questions, implementation details, or for other discussions, please open an issue or PR on GitHub
