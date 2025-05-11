#!/usr/bin/env python3
"""
brainstemx.qa  –  pipeline QA / validation in pure Python

Checks:
  • File presence
  • Basic image statistics (voxels, min/max, mean, std)
  • Threshold-based sanity for masks
  • Dice / Jaccard between brainstem_mask & atlas (if atlas present)
  • CC (Pearson) between flair_to_t1 and t1_brain
Outputs:
  results/IDxx/qa_log.txt
  results/IDxx/qa_report.json
Return codes:
  0 = COMPLETE   (all present + all valid)
  1 = INVALID    (all present but one+ check failed)
  2 = INCOMPLETE (missing files)
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import label
from scipy.stats import pearsonr

EXPECTED = {
    "t1_brain.nii.gz":              dict(min_nonzero=10_000),
    "flair_to_t1.nii.gz":           {},
    "brainstem_mask.nii.gz":        dict(min_nonzero=1_000, max_nonzero=50_000),
    "lesion_sd2.0.nii.gz":          dict(min_nonzero=4),
    "QC_overlay.png":               {},          # just presence
}

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"

# ───────────────────────── helpers ────────────────────────────────────────────
def load_nii(path: Path):
    return nib.load(str(path))

def img_stats(img):
    data = img.get_fdata()
    nz   = data[data != 0]
    return dict(nonzero=len(nz),
                min=float(data.min()),
                max=float(data.max()),
                mean=float(data.mean()),
                std=float(data.std()))

def mask_threshold_check(stats, min_nz=None, max_nz=None):
    ok = True
    if min_nz is not None and stats["nonzero"] < min_nz: ok = False
    if max_nz is not None and stats["nonzero"] > max_nz: ok = False
    return ok

def dice(mask1, mask2):
    a = mask1.get_fdata() > 0
    b = mask2.get_fdata() > 0
    inter = (a & b).sum()
    return 2*inter / (a.sum() + b.sum() + 1e-5)

def jaccard(mask1, mask2):
    a = mask1.get_fdata() > 0
    b = mask2.get_fdata() > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / (union + 1e-5)

def cc(img1, img2, mask=None):
    a = img1.get_fdata().flatten()
    b = img2.get_fdata().flatten()
    if mask is not None:
        m = mask.get_fdata().flatten() > 0
        a, b = a[m], b[m]
    return pearsonr(a, b)[0]

# ───────────────────────── main QA routine ────────────────────────────────────
def qa_subject(subj_dir: Path) -> int:
    log_file = subj_dir/"qa_log.txt"
    logging.basicConfig(filename=log_file, level=logging.INFO, format=LOG_FMT)
    lg = logging.getLogger("QA")
    lg.info("QA start for %s", subj_dir.name)
    report = {"subject": subj_dir.name, "files": {}, "metrics": {}}

    all_present, all_valid = True, True

    # presence + basic stats
    for rel, cfg in EXPECTED.items():
        fp = subj_dir/rel
        entry = {"present": fp.exists()}
        if not fp.exists():
            lg.error("MISSING %s", rel); all_present=False
            report["files"][rel]=entry; continue
        if fp.suffix == ".gz":
            img = load_nii(fp); st = img_stats(img)
            entry.update(st)
            if not mask_threshold_check(st, **cfg):
                lg.error("INVALID %s – nz=%d", rel, st["nonzero"])
                entry["status"]="INVALID"; all_valid=False
            else:
                entry["status"]="OK"
        else:
            entry["status"]="OK"
        report["files"][rel]=entry

    # advanced metrics only if all key files present
    if all_present:
        flair = load_nii(subj_dir/"flair_to_t1.nii.gz")
        t1    = load_nii(subj_dir/"t1_brain.nii.gz")
        bs    = load_nii(subj_dir/"brainstem_mask.nii.gz")

        cc_val = cc(flair, t1, bs)
        report["metrics"]["flair_t1_cc"] = round(float(cc_val),3)
        lg.info("CC(FLAIR,T1)=%.3f", cc_val)
        if cc_val < 0.2: all_valid=False

        # Dice / Jaccard with atlas if available
        atlas = subj_dir.parent.parent/"atlases"/"brainstem_atlas_in_native.nii.gz"
        if atlas.exists():
            d = dice(bs, load_nii(atlas))
            j = jaccard(bs, load_nii(atlas))
            report["metrics"]["brainstem_dice"]   = round(float(d),3)
            report["metrics"]["brainstem_jaccard"]= round(float(j),3)
            lg.info("Dice brainstem vs atlas = %.3f", d)

    # summary
    if all_present and all_valid:
        status = 0; summary = "COMPLETE"
    elif all_present and not all_valid:
        status = 1; summary = "INVALID"
    else:
        status = 2; summary = "INCOMPLETE"
    lg.info("QA summary: %s", summary)
    report["summary"] = summary

    (subj_dir/"qa_report.json").write_text(json.dumps(report, indent=2))
    return status

# ───────────────────────── CLI wrapper ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subj", required=True, help="results/IDxx folder")
    ret = qa_subject(Path(ap.parse_args().subj))
    exit(ret)

if __name__ == "__main__":
    main()

