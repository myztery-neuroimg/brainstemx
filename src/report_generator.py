"""
brainstemx.report_generator – GPT-4.1 vision → .docx

Requirements:
    pip install openai>=1.5 python-docx pandas pillow
"""

from __future__ import annotations
import base64, argparse
from datetime import date
from pathlib import Path
import pandas as pd, openai, docx
from docx.shared import Inches
import nibabel as nib, numpy as np, matplotlib.pyplot as plt

MODEL = "gpt-4.1-vision-preview"

# ---------- helpers ----------------------------------------------------------
def to_b64(p:Path)->str:
    return "data:image/png;base64,"+base64.b64encode(p.read_bytes()).decode()

def sagittal(subj:Path)->Path:
    out=subj/"sag_overlay.png"
    if out.exists(): return out
    flair=nib.load(str(subj/"flair_to_t1.nii.gz")).get_fdata()
    mask =nib.load(str(subj/"lesion_sd2.0.nii.gz")).get_fdata()
    x=flair.shape[0]//2
    norm=(flair[x,:,:]-flair.min())/flair.ptp()
    rgba=np.stack([norm,norm,norm,np.ones_like(norm)],2)
    rgba[mask[x,:,:]>0]=[1,0,0,0.6]
    plt.imsave(out,(rgba*255).astype(np.uint8))
    return out

def collect_imgs(subj)->tuple[list[Path],list[dict]]:
    files=[subj/"QC_overlay.png", sagittal(subj)] + sorted(subj.glob("qc_*.png"))
    files=[p for p in files if p.exists()][:4]
    blocks=[{"type":"image_url","image_url":{"url":to_b64(p)}} for p in files]
    return files,blocks

def build_messages(csv_txt:str, blocks:list[dict])->list[dict]:
    system="You are a board-certified neuroradiologist. Write MRI brain-stem Findings & Impression."
    user=[{"type":"text","text":f"Lesion metrics CSV:\n```csv\n{csv_txt.strip()}\n```"}]+blocks
    return [{"role":"system","content":system},
            {"role":"user","content":user}]

def gpt_dictation(key:str,msgs)->str:
    openai.api_key=key
    res=openai.chat.completions.create(model=MODEL,messages=msgs,
                                       max_tokens=400,response_format={"type":"text"})
    return res.choices[0].message.content.strip()

def docx_report(subj:Path,text:str,imgs:list[Path])->Path:
    doc=docx.Document()
    doc.add_heading(f"BrainStemX auto-report – {subj.name}",1)
    doc.add_paragraph(f"Date: {date.today().isoformat()}")
    for line

