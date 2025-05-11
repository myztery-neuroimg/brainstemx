"""
brainstemx.core · v1.2
Utility layer: config, logging, diagnostics, preprocessing, registration,
mask generation, lesion detection, clustering, QC overlay.
"""

from __future__ import annotations
import os, sys, json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, List
import numpy as np, pandas as pd, ants
from scipy.ndimage import binary_opening, generate_binary_structure
from skimage.measure import label, regionprops_table
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    n_jobs:int = min(os.cpu_count(),22); overwrite:bool=False; resume:bool=True
    log_level:str="INFO"

    brain_template:str="${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz"
    brain_mask_tpl:str="${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz"
    flair_template:str="${FSLDIR}/data/standard/MNI152_T2_FLAIR_1mm.nii.gz"
    bs_mask_tpl:str="${FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-brainstem.nii.gz"
    cit168_tpl:str="${FSLDIR}/data/atlases/CIT168/CIT168_relabels.nii.gz"

    skullstrip_order:Sequence[str]=("ants",)
    n4_shrink:int=2; flair_n4_shrink:int=1

    thresholds_sd:Sequence[float]=(1.5,2.0,2.5,3.0); primary_sd:float=2.0
    min_vox:int=5; open_radius:int=1; connectivity:int=26
    max_elongation:float=12.0

    def resolve(self)->None:
        for k,v in vars(self).items():
            if isinstance(v,str) and v.startswith("${"):
                setattr(self,k,os.path.expandvars(v))
    load = classmethod(lambda cls,p: cls()._load(p))
    def _load(self,p:Path):
        self.__dict__.update(json.loads(p.read_text())); self.resolve(); return self
    def save(self,p:Path): self.resolve(); p.write_text(json.dumps(self.__dict__,indent=2))

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def get_logger(name:str,fp:Path,level:str)->logging.Logger:
    lg=logging.getLogger(name)
    if lg.handlers: return lg
    lg.setLevel(level)
    fmt=logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh=logging.FileHandler(fp,mode="a"); fh.setFormatter(fmt)
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    lg.addHandler(fh); lg.addHandler(sh); return lg

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics helpers
# ──────────────────────────────────────────────────────────────────────────────
def log_image_stats(img:"ants.ANTsImage",tag:str,lg)->dict:
    arr=img.numpy(); stats=dict(dtype=str(arr.dtype),shape=list(arr.shape),
        spacing=[round(s,3) for s in img.spacing],
        min=float(arr.min()),max=float(arr.max()),
        mean=float(arr.mean()),std=float(arr.std()))
    lg.debug("%s shape=%s spacing=%s min/mean/std/max= %.2f/%.2f/%.2f/%.2f",
             tag,stats['shape'],stats['spacing'],
             stats['min'],stats['mean'],stats['std'],stats['max'])
    return stats

def write_img(img:"ants.ANTsImage",fp:Path,label:str,lg,audit:dict,mask=False):
    img=img.clone().astype("uint8" if mask else "float32")
    ants.image_write(img,fp.as_posix())
    audit["files"][label]=dict(path=str(fp),**log_image_stats(img,label,lg))

def resample_if_needed(mov:"ants.ANTsImage",ref:"ants.ANTsImage",lg,tol=0.10):
    spm,spr=np.array(mov.spacing),np.array(ref.spacing)
    if np.max(abs(spm-spr)/spr)>tol:
        lg.warning("Resampling %s→%s mm",spm.round(3),spr.round(3))
        mov=ants.resample_image(mov,ref.shape,use_voxels=True,interp_type=1)
    return mov

def check_mask(mask:"ants.ANTsImage",min_vox:int,lg,label:str,audit:dict):
    vox=int(mask.numpy().sum()); flag="OK"
    if vox==0: flag="EMPTY"; lg.warning("%s empty",label)
    elif vox<min_vox: flag="SMALL"; lg.warning("%s tiny (%d vox)",label,vox)
    audit.setdefault("checks",{})[label]=dict(vox=vox,status=flag)

# ──────────────────────────────────────────────────────────────────────────────
# Skull-strip, preprocessing, registration, masks  (functions unchanged from
# previous response but shortened here for brevity – copy the full v1.2 code).
# ──────────────────────────────────────────────────────────────────────────────
# include skullstrip, preprocess_flair, preprocess_generic, register_to_t1,
# brainstem_masks, wm_segmentation, lesion_masks, cluster_metrics, qc_overlay
# ... (use full versions from previous answer)

