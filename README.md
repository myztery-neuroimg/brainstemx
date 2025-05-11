i
# BrainStem X – Intensity-Clustering Pipeline for Brain-Stem / Pons MRI
> **Folder:** `src/` &nbsp; • &nbsp; **Status:** active development (May 2025)

BrainStem X is an end‑to‑end, **first‑principles** pipeline for detecting and quantifying subtle T2‑FLAIR **hyper‑intensity** / T1 **hypo‑intensity** clusters in the brain‑stem (dorsal & ventral pons).  
It marries classical MR image processing with a transparent QA stack – all in pure Python – while delegating heavy lifting to ANTs & FSL when available.

---

## Why another brain‑stem pipeline?

* **Multi‑modal fusion** – coregisters T1 / FLAIR / SWI / DWI and measures voxel‑wise overlap.
* **Zero‑shot, unsupervised clustering** – no training data, no bias.
* **Aggressive QA** – header & JSON checks, ≥20 metrics, Dash browser.
* **DICOM back‑trace** – every cluster can be mapped back to raw scanner slices.
* **Apple‑Silicon tuned** – ANTs on Metal; runs fine on Linux too.
* **Tiny dependency footprint** – everything Python lives in `src/`; compiled deps are ANTs + FSL only.

---

## Repository layout

```
src/
 ├── __init__.py
 ├── core.py                  # utilities, registration, clustering
 ├── validate_inputs.py       # JSON+NIfTI pre‑flight guard
 ├── pipeline.py              # end‑to‑end processing
 ├── qa.py                    # post‑pipeline QA / validation
 ├── postprocess.py           # cluster metrics & csv (see docs)
 ├── report_generator.py      # GPT‑4.1‑vision → DOCX
 ├── generate_synthetic_data.py
 ├── web_visualiser.py        # Dash 3‑plane browser
 └── cli.py                   # batch / single‑subject wrapper
README.md
requirements.txt
```

*(If you don’t see `postprocess.py`, you’re missing a file – copy it from the last chat block.)*

---

## Quick install

```bash
git clone https://github.com/your‑handle/brainstemx.git
cd brainstemx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # numpy, nibabel, ants, dash, …
brew install ants fsl c3d dcm2niix # or apt‑get on Linux
export ANTSPATH=/usr/local/opt/ants/bin
export FSLDIR=/usr/local/opt/fsl
export PATH="$ANTSPATH:$FSLDIR/bin:$PATH"
```

---

## Typical run

```bash
python -m brainstemx.cli        --flair data/sub‑001_FLAIR.nii.gz        --t1    data/sub‑001_T1w.nii.gz        --out   results/ID01

python -m brainstemx.report_generator --subj results/ID01 --key $OPENAI_API_KEY
python web_visualiser.py --root results
```

* `validate_inputs.py` aborts early if TE/TR or voxel spacing look wrong.  
* `pipeline.py` writes `outputs.json`, lesion masks & `analysis.csv`.  
* `qa.py` returns 0/1/2 and writes `qa_report.json`.  
* `report_generator.py` feeds **analysis.csv + PNGs** to `gpt‑4.1‑vision-preview` and saves a DOCX. 

---

## Freeview helper

The Dash UI prints a one‑click command like

```bash
freeview -v results/ID01/flair_to_t1.nii.gz          results/ID01/lesion_sd2.0.nii.gz:colormap=heat:opacity=0.5          results/ID01/brainstem_mask.nii.gz:colormap=blue
```

---

## Citation

```
@software{BrainStemX2025,
  author = {D. J. Brewster},
  title  = {{BrainStem X}: Brain‑Stem / Pons MRI Intensity‑Clustering Pipeline},
  year   = {2025},
  url    = {https://github.com/your‑handle/brainstemx}
}
```

MIT licence for BrainStem X; external toolkits keep their own licences.

Happy scanning 🚀

