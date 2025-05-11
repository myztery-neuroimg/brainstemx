"""
brainstemx.postprocess – quantitative analysis of lesion clusters.

Reads:
  * flair_to_t1.nii.gz
  * (optional) t1_brain.nii.gz, dwi_to_t1.nii.gz, swi_to_t1.nii.gz
  * lesion_mask_sdX.nii.gz  (any SDs your pipeline produced)
  * brainstem_mask.nii.gz

Outputs:
  analysis.csv    per-cluster metrics
  overlap.json    voxel-overlap matrix
  postprocess.log verbose log
"""

import logging, json, csv
from pathlib import Path
import numpy as np, ants, pandas as pd
from skimage.measure import label
from datetime import datetime

def _log_img_stats(img, tag, lg):
    arr = img.numpy()
    lg.debug("%s  min/mean/p90/max = %.2f / %.2f / %.2f / %.2f",
             tag, arr.min(), arr.mean(), np.percentile(arr,90), arr.max())

def analyse(subject_dir: Path, primary_sd: float = 2.0):
    log = logging.getLogger("postprocess")
    fh  = logging.FileHandler(subject_dir/"postprocess.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    log.addHandler(fh); log.setLevel("INFO")

    flair = ants.image_read((subject_dir/"flair_to_t1.nii.gz").as_posix())
    t1    = ants.image_read((subject_dir/"t1_brain.nii.gz").as_posix()) if (subject_dir/"t1_brain.nii.gz").exists() else None
    swi   = ants.image_read((subject_dir/"swi_to_t1.nii.gz").as_posix()) if (subject_dir/"swi_to_t1.nii.gz").exists() else None
    bs    = ants.image_read((subject_dir/"brainstem_mask.nii.gz").as_posix())

    # ---- gather lesion masks --------------------
    lesion_masks = {sd: ants.image_read((subject_dir/f"lesion_sd{sd:.1f}.nii.gz").as_posix())
                    for sd in [primary_sd]}

    # ---- per-cluster metrics --------------------
    primary = lesion_masks[primary_sd].numpy().astype(bool)
    lbl, n  = label(primary, return_num=True, connectivity=2)
    voxvol  = np.prod(flair.spacing)
    rows    = []
    for idx in range(1, n+1):
        cl = (lbl==idx)
        vox = int(cl.sum())
        if vox==0: continue
        mm3 = vox*voxvol
        flair_vals = flair.numpy()[cl]
        p90 = np.percentile(flair_vals,90)
        mean= flair_vals.mean()
        row = dict(cluster=idx, voxels=vox, mm3=round(mm3,2),
                   flair_mean=round(mean,2), flair_p90=round(p90,2))
        if t1 is not None:
            t1_vals = t1.numpy()[cl]
            row.update(t1_mean=round(t1_vals.mean(),2),
                       t1_p10 =round(np.percentile(t1_vals,10),2))
        if swi is not None:
            swi_vals = swi.numpy()[cl]
            row.update(swi_min=round(swi_vals.min(),2))
        rows.append(row)
        log.info("cluster %d: vox=%d  mm3=%.1f  flair_mean=%.2f  p90=%.2f",
                 idx, vox, mm3, mean, p90)

    csv_path = subject_dir/"analysis.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ---- overlap matrix -------------------------
    overlap = {}
    for a_name, a_mask in lesion_masks.items():
        a = a_mask.numpy().astype(bool)
        overlap[str(a_name)] = {}
        if t1 is not None:
            hypo = (t1.numpy() < np.percentile(t1.numpy()[bs.numpy()>0],10))
            overlap[str(a_name)]["T1_hypo_vox"] = int((a & hypo).sum())
        if swi is not None:
            bloom = (swi.numpy() < np.percentile(swi.numpy()[bs.numpy()>0],5))
            overlap[str(a_name)]["SWI_bloom_vox"] = int((a & bloom).sum())
    (subject_dir/"overlap.json").write_text(json.dumps(overlap, indent=2))

    log.info("Finished post-processing (%d clusters)", len(rows))
    return csv_path

