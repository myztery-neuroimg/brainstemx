#!/usr/bin/env python3
"""
brainstemx.validate_inputs – pre-flight guard for NIfTI + side-car JSON.

Run automatically at the start of process_subject()
Raises RuntimeError on critical failure.
Creates inputs_valid.json inside subject folder.
"""

from __future__ import annotations
import json, logging, textwrap
from pathlib import Path
from typing import Tuple
import nibabel as nib
import numpy as np

CRITICAL_KEYS = ["RepetitionTime", "EchoTime", "MagneticFieldStrength"]
RANGES = {
    "FLAIR": dict(EchoTime=(0.08, 0.15), RepetitionTime=(6.0, 11.0)),
    "T1":    dict(EchoTime=(0.002, 0.008), RepetitionTime=(1.3, 3.0)),
}

def read_json(sidecar: Path) -> dict:
    try:
        return json.loads(sidecar.read_text())
    except Exception as e:
        raise RuntimeError(f"Malformed JSON side-car: {sidecar} ({e})")

def check_range(val: float, lo: float, hi: float) -> bool:
    return lo <= val <= hi

def validate_pair(nii: Path, side: Path, modality: str, lg) -> dict:
    hdr = nib.load(str(nii)).header
    report = {"file": nii.name, "modality": modality, "status": "PASS"}

    # ---- header vs JSON voxel spacing ------------------------------------
    if side.exists():
        js = read_json(side)
        voxs = np.round(hdr.get_zooms()[:3], 4)
        if "PixelSpacing" in js and "SliceThickness" in js:
            json_voxs = js["PixelSpacing"] + [js["SliceThickness"]]
            if not np.allclose(voxs, json_voxs, atol=0.01):
                report["status"] = "FAIL"
                report["error"] = f"Voxel size mismatch header={voxs} json={json_voxs}"
                lg.error(report["error"])

        # ---- critical keys present? --------------------------------------
        for k in CRITICAL_KEYS:
            if k not in js:
                report["status"] = "FAIL"
                report.setdefault("missing", []).append(k)

        # ---- physiologic range check -------------------------------------
        rngs = RANGES.get(modality.upper())
        if rngs:
            for key,(lo,hi) in rngs.items():
                if key in js and not check_range(js[key], lo, hi):
                    report["status"]="FAIL"
                    msg=f"{key}={js[key]:.3f} outside [{lo},{hi}]"
                    report.setdefault("violations",[]).append(msg); lg.error(msg)
    else:
        lg.warning("No JSON side-car for %s", nii.name)
        report["warning"] = "no_sidecar"

    return report

def validate_inputs(subj_dir: Path, flair_p: Path, t1_p: Path, lg, dwi_p: Path = None, swi_p: Path = None) -> None:
    """Validate all input files including optional modalities.
    
    Args:
        subj_dir: Output directory for the subject
        flair_p: Path to FLAIR NIfTI file (required)
        t1_p: Path to T1 NIfTI file (optional but recommended)
        lg: Logger object
        dwi_p: Path to DWI NIfTI file (optional)
        swi_p: Path to SWI NIfTI file (optional)
    
    Raises:
        RuntimeError: If any validation check fails
    """
    checks = []
    
    # Required FLAIR validation
    checks.append(validate_pair(flair_p, flair_p.with_suffix(".json"), "FLAIR", lg))
    
    # Optional T1 validation
    if t1_p and t1_p.exists():
        checks.append(validate_pair(t1_p, t1_p.with_suffix(".json"), "T1", lg))
    
    # Optional DWI validation
    if dwi_p and dwi_p.exists():
        checks.append(validate_pair(dwi_p, dwi_p.with_suffix(".json"), "DWI", lg))
    
    # Optional SWI validation
    if swi_p and swi_p.exists():
        checks.append(validate_pair(swi_p, swi_p.with_suffix(".json"), "SWI", lg))

    # Write validation report
    subj_report = subj_dir/"inputs_valid.json"
    subj_report.write_text(json.dumps(checks, indent=2))

    # Fail fast if any critical error
    for c in checks:
        if c["status"] == "FAIL":
            raise RuntimeError(f"Input validation failed for {c['file']}")

# ---------------------------------------------------------------- CLI ---------
if __name__ == "__main__":
    import argparse, sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--flair", required=True, help="FLAIR NIfTI image (required)")
    ap.add_argument("--t1", help="T1 NIfTI image (optional but recommended)")
    ap.add_argument("--dwi", help="DWI NIfTI image (optional)")
    ap.add_argument("--swi", help="SWI NIfTI image (optional)")
    ap.add_argument("--out", required=True, help="Output directory")
    a = ap.parse_args()
    
    validate_inputs(
        Path(a.out),
        Path(a.flair),
        Path(a.t1) if a.t1 else None,
        logging.getLogger("validate"),
        Path(a.dwi) if a.dwi else None,
        Path(a.swi) if a.swi else None
    )

