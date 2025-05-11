#!/usr/bin/env python3
"""
Synthetic test-case generator for BrainStemX.

✔ 5 lesion variants per run (irregular, graded intensity)
✔ Uses existing brainstem_mask.nii.gz when present
✔ Adds optional hypointense “black-hole” on T1
✔ QC PNG overlay
"""

from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, nibabel as nib, ants
from scipy.ndimage import gaussian_filter, binary_dilation
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------- helpers
def _load(path: Path, float32=True):
    img = ants.image_read(path.as_posix())
    return img.astype("float32") if float32 else img

def _save(img, path):
    ants.image_write(img, path.as_posix())

def _brainstem_mask(auto_mask: Path|None, t1_img):
    if auto_mask and auto_mask.exists():
        return _load(auto_mask, float32=False)
    # crude fallback: lower half of brain & low-intensity voxels
    arr = t1_img.numpy()
    rough = (arr < np.percentile(arr, 30)).astype(np.uint8)
    rough[:,:,:arr.shape[2]//2] = 0
    return ants.from_numpy(rough, origin=t1_img.origin,
                           spacing=t1_img.spacing, direction=t1_img.direction)

def _irregular_blob(center, shape):
    """Make Perlin-like irregular lesion."""
    zz,yy,xx=np.indices(shape)
    dist = np.sqrt((xx-center[0])**2 +
                   (yy-center[1])**2 +
                   (zz-center[2])**2)
    base = np.exp(-(dist**2)/(2*(random.uniform(2,5)**2)))
    noise= gaussian_filter(np.random.rand(*shape), sigma=3)
    blob = base + 0.3*noise
    thresh= np.percentile(blob, 65)
    mask = (blob>thresh).astype(np.uint8)
    # random dilation for spiculation
    for _ in range(random.randint(1,3)):
        mask=binary_dilation(mask, iterations=1).astype(np.uint8)
    return mask

def _qc(base, mask, out_png, title):
    z=base.shape[2]//2
    plt.figure(figsize=(5,5))
    plt.imshow(base[:,:,z].T,cmap="gray",origin="lower")
    plt.imshow(np.ma.masked_where(mask[:,:,z]==0,mask[:,:,z]).T,
               cmap="autumn",alpha=0.6,origin="lower")
    plt.axis("off"); plt.title(title); plt.savefig(out_png,dpi=140,bbox_inches="tight"); plt.close()


# ----------------------------------------------------------- main generation
def generate(src_t1, src_flair, bs_mask_p, out_dir,
             seed=13, cases=5, qc=False):
    np.random.seed(seed); random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_img   = _load(src_t1)
    flair_in = _load(src_flair)
    bs_mask  = _brainstem_mask(bs_mask_p, t1_img)

    # copy original images
    _save(t1_img,   out_dir/"t1_native.nii.gz")
    _save(flair_in, out_dir/"flair_native.nii.gz")

    coords = np.column_stack(np.where(bs_mask.numpy()>0))
    for idx in range(1, cases+1):
        tag=f"{idx:02d}"
        # pick random center & build lesion
        cx,cy,cz = coords[np.random.randint(0,len(coords))]
        mask_np  = _irregular_blob((cx,cy,cz), flair_in.shape)
        lesion   = ants.from_numpy(mask_np, origin=flair_in.origin,
                                   spacing=flair_in.spacing,
                                   direction=flair_in.direction)
        # graded intensity: core brighter
        dist = ants.iMath(lesion, "GD")
        rim_factor = 0.6+0.4*(dist.numpy()/dist.numpy().max())
        flair_syn  = flair_in.clone()
        flair_syn.numpy()[mask_np==1] *= rim_factor[mask_np==1]*(random.uniform(1.3,1.8))

        # optional T1 hypointensity 50 % of the time
        t1_syn = t1_img.clone()
        if random.random()<0.5:
            t1_syn.numpy()[mask_np==1] *= random.uniform(0.5,0.8)

        # save
        _save(flair_syn, out_dir/f"flair_synth_{tag}.nii.gz")
        _save(lesion,    out_dir/f"lesion_mask_{tag}.nii.gz")
        _save(t1_syn,    out_dir/f"t1_synth_{tag}.nii.gz")

        audit = dict(center=[int(cx),int(cy),int(cz)],
                     voxels=int(mask_np.sum()),
                     hypointense=bool("t1_synth" in locals()))
        (out_dir/f"audit_{tag}.json").write_text(json.dumps(audit, indent=2))

        if qc:
            _qc(flair_syn.numpy(), mask_np, out_dir/f"qc_{tag}.png", f"Synth {tag}")

        print(f"[✓] case {tag}: vox={audit['voxels']}  core=({cx},{cy},{cz})")

    print("\nOpen a case in Freeview:")
    print(f"freeview -v {out_dir}/flair_synth_01.nii.gz "
          f"{out_dir}/lesion_mask_01.nii.gz:colormap=heat:opacity=0.5 "
          f"{out_dir}/t1_synth_01.nii.gz")


# --------------------------------------------------------------------------- CL
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1",    required=True)
    ap.add_argument("--flair", required=True)
    ap.add_argument("--mask")
    ap.add_argument("--out",   required=True)
    ap.add_argument("--cases", type=int, default=5)
    ap.add_argument("--seed",  type=int, default=17)
    ap.add_argument("--qc",    action="store_true")
    a=ap.parse_args()
    generate(Path(a.t1), Path(a.flair),
             Path(a.mask) if a.mask else None,
             Path(a.out), a.seed, a.cases, a.qc)

