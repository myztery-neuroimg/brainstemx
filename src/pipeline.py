"""
brainstemx.pipeline · v1.2
Full pipeline with verbose step logging & audit JSON.
"""

from __future__ import annotations
import json, traceback, sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import ants
from joblib import Parallel, delayed

from .core import (Config,get_logger,write_img,check_mask,resample_if_needed,
                   skullstrip,preprocess_flair,preprocess_generic,
                   register_to_t1,brainstem_masks,
                   wm_segmentation,lesion_masks,cluster_metrics,qc_overlay)
from .validate_inputs import validate_inputs

# ───────────────────────────────────────────────
def process_subject(subj_id:str, flair_p:Path, t1_p:Optional[Path],
                    out:Path, cfg:Config,
                    dwi_p:Optional[Path]=None, swi_p:Optional[Path]=None):
    """Run full BrainStemX processing on one subject directory."""
    out.mkdir(parents=True,exist_ok=True)
    lg=get_logger(subj_id,out/"pipeline.log",cfg.log_level)
    audit={"subject":subj_id,"started":datetime.utcnow().isoformat(),"files":{}}

    try:
        # STEP 0a - Validate inputs
        lg.info("STEP 0a ·  validate input files")
        validate_inputs(out, flair_p, t1_p, lg, dwi_p, swi_p)
        
        # STEP 0b - Preprocess FLAIR
        lg.info("STEP 0b ·  load & preprocess FLAIR")
        flair_in=ants.image_read(flair_p.as_posix()).astype("float32")
        flair_pp=preprocess_flair(flair_in,cfg,lg)
        write_img(flair_pp,out/"flair_pp.nii.gz","flair_pp",lg,audit)

        # STEP 1
        lg.info("STEP 1  ·  skull-strip")
        t1_img=ants.image_read(t1_p.as_posix()).astype("float32") if t1_p and t1_p.exists() else None
        brain_t1,mask_t1=skullstrip(t1_img or flair_pp,cfg,lg)
        write_img(brain_t1,out/"t1_brain.nii.gz","t1_brain",lg,audit)
        write_img(mask_t1,out/"brain_mask.nii.gz","brain_mask",lg,audit,mask=True)
        check_mask(mask_t1,5000,lg,"brain_mask",audit)

        # STEP 2
        lg.info("STEP 2  ·  register FLAIR → T1")
        flair_pp=resample_if_needed(flair_pp,brain_t1,lg)
        flair_reg,fwd,inv=register_to_t1(brain_t1,flair_pp,cfg,lg,"FLAIR")
        write_img(flair_reg,out/"flair_to_t1.nii.gz","flair_to_t1",lg,audit)

        # STEP 2b  (optional modalities)
        for tag,pp in {"DWI":dwi_p,"SWI":swi_p}.items():
            if pp and pp.exists():
                lg.info("STEP 2b ·  register %s",tag)
                mod=preprocess_generic(ants.image_read(pp.as_posix()),lg)
                mod=resample_if_needed(mod,brain_t1,lg)
                mod_r,*_=register_to_t1(brain_t1,mod,cfg,lg,tag)
                write_img(mod_r,out/f"{tag.lower()}_to_t1.nii.gz",f"{tag.lower()}_to_t1",lg,audit)

        # STEP 3
        lg.info("STEP 3  ·  brain-stem mask")
        bs,dorsal,ventral=brainstem_masks(brain_t1,inv,cfg,lg)
        write_img(bs,out/"brainstem_mask.nii.gz","brainstem_mask",lg,audit,mask=True)
        check_mask(bs,2000,lg,"brainstem_mask",audit)

        # STEP 4
        lg.info("STEP 4  ·  WM segmentation & lesion masks")
        wm_mask,_=wm_segmentation(brain_t1,flair_reg,mask_t1,lg)
        write_img(wm_mask,out/"wm_mask.nii.gz","wm_mask",lg,audit,mask=True)
        masks=lesion_masks(flair_reg,wm_mask,bs,cfg,lg)
        for s,m in masks.items():
            write_img(m,out/f"lesion_sd{s:.1f}.nii.gz",f"lesion_sd{s:.1f}",lg,audit,mask=True)
        primary=masks[cfg.primary_sd]; check_mask(primary,cfg.min_vox,lg,"lesion_primary",audit)

        # STEP 5
        lg.info("STEP 5  ·  clustering metrics")
        csv=cluster_metrics(primary,cfg,out,lg)
        audit["files"]["clusters_csv"]=str(csv)

        # STEP 6
        lg.info("STEP 6  ·  QC overlay")
        qc_overlay(flair_reg,primary,out/"QC_overlay.png","FLAIR + lesions")
        audit["files"]["QC_overlay"]=str(out/"QC_overlay.png")

    except Exception as e:
        lg.error("pipeline failure: %s",e); lg.debug(traceback.format_exc())
        audit["error"]=str(e)

    audit["finished"]=datetime.utcnow().isoformat()
    (out/"outputs.json").write_text(json.dumps(audit,indent=2))
    lg.info("Finished subject %s",subj_id)

# ───────────────────────────────────────────────
def process_batch(tsv:Path,cfg:Config):
    rows=tsv.read_text().strip().splitlines()
    Parallel(n_jobs=cfg.n_jobs)(delayed(_dispatch)(row,cfg) for row in rows)

def _dispatch(row:str,cfg:Config):
    sid,flair,t1,*rest=row.split("\t")+["None"]*5
    dwi,swi,out=rest[:3]
    out_dir=Path(out) if out!="None" else Path("results")/sid
    process_subject(sid,Path(flair),Path(t1)if t1!="None"else None,
                    out_dir,cfg,Path(dwi) if dwi!="None" else None,
                    Path(swi) if swi!="None" else None)

