"""
brainstemx.core · v1.2
Utility layer: config, logging, diagnostics, preprocessing, registration,
mask generation, lesion detection, clustering, QC overlay.
"""

from __future__ import annotations
import os, sys, json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, List, Optional
import numpy as np, pandas as pd, ants
from scipy.ndimage import binary_opening, generate_binary_structure
from skimage.measure import label, regionprops_table
import matplotlib.pyplot as plt

# Try to import SynthStrip for fallback skull-stripping
try:
    from synthstrip import SynthStrip
    SYNTHSTRIP_AVAILABLE = True
except ImportError:
    SYNTHSTRIP_AVAILABLE = False
    logging.getLogger(__name__).warning("SynthStrip not available - install with 'pip install synthstrip'")

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
    # Note: CIT168 is not currently used but kept for potential future subcortical segmentation
    cit168_tpl:str="${FSLDIR}/data/atlases/CIT168/CIT168_relabels.nii.gz"

    skullstrip_order:Sequence[str]=("ants", "synthstrip")
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
    """Set up a logger with file and console output.
    
    Args:
        name: Logger name
        fp: Path for the log file
        level: Logging level (e.g., "INFO", "DEBUG")
        
    Returns:
        Configured logger instance
    """
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
    """Log and return basic image statistics.
    
    Args:
        img: ANTs image to analyze
        tag: Label for the image in log messages
        lg: Logger instance
        
    Returns:
        Dictionary of image statistics
    """
    arr=img.numpy(); stats=dict(dtype=str(arr.dtype),shape=list(arr.shape),
        spacing=[round(s,3) for s in img.spacing],
        min=float(arr.min()),max=float(arr.max()),
        mean=float(arr.mean()),std=float(arr.std()))
    lg.debug("%s shape=%s spacing=%s min/mean/std/max= %.2f/%.2f/%.2f/%.2f",
             tag,stats['shape'],stats['spacing'],
             stats['min'],stats['mean'],stats['std'],stats['max'])
    return stats

def write_img(img:"ants.ANTsImage",fp:Path,label:str,lg,audit:dict,mask=False):
    """Write an image to disk and record metadata in the audit log.
    
    Args:
        img: ANTs image to write
        fp: Path to write the image to
        label: Label for the image in the audit log
        lg: Logger instance
        audit: Audit dictionary to update
        mask: If True, convert to uint8 mask, otherwise float32
    """
    img=img.clone().astype("uint8" if mask else "float32")
    ants.image_write(img,fp.as_posix())
    audit["files"][label]=dict(path=str(fp),**log_image_stats(img,label,lg))

def resample_if_needed(mov:"ants.ANTsImage",ref:"ants.ANTsImage",lg,tol=0.10):
    """Resample an image to match a reference if spacing differs significantly.
    
    Args:
        mov: Moving (input) image to resample if needed
        ref: Reference image with target spacing
        lg: Logger instance
        tol: Tolerance fraction for spacing differences
        
    Returns:
        Resampled image or original if resampling not needed
    """
    spm,spr=np.array(mov.spacing),np.array(ref.spacing)
    if np.max(abs(spm-spr)/spr)>tol:
        lg.warning("Resampling %s→%s mm",spm.round(3),spr.round(3))
        mov=ants.resample_image(mov,ref.shape,use_voxels=True,interp_type=1)
    return mov

def check_mask(mask:"ants.ANTsImage",min_vox:int,lg,label:str,audit:dict):
    """Check if a mask is valid (non-empty and large enough).
    
    Args:
        mask: Mask image to check
        min_vox: Minimum number of voxels required
        lg: Logger instance
        label: Label for the mask in logs and audit
        audit: Audit dictionary to update
    """
    vox=int(mask.numpy().sum()); flag="OK"
    if vox==0: flag="EMPTY"; lg.warning("%s empty",label)
    elif vox<min_vox: flag="SMALL"; lg.warning("%s tiny (%d vox)",label,vox)
    audit.setdefault("checks",{})[label]=dict(vox=vox,status=flag)

def check_file_dependencies(required_files: List[Path], lg, context="") -> bool:
    """Verify all required files exist before proceeding.
    
    Args:
        required_files: List of Path objects that must exist
        lg: Logger object
        context: String describing the operation requiring these files
        
    Returns:
        True if all files exist
        
    Raises:
        FileNotFoundError: If any required files are missing
    """
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        msg = f"Missing required files for {context}: {', '.join(missing)}"
        lg.error(msg)
        raise FileNotFoundError(msg)
    return True

# -----------------------------------------------------------------------------#
#  Skull-strip (ANTs with SynthStrip fallback)
# -----------------------------------------------------------------------------#

def skullstrip(img: "ants.ANTsImage",
               cfg: Config,
               lg) -> Tuple["ants.ANTsImage", "ants.ANTsImage"]:
    """Skull-strip a brain image with multiple fallback options.
    
    Tries methods in this order:
    1. ANTs (if in skullstrip_order)
    2. SynthStrip (if available and in skullstrip_order or ANTs fails)
    
    Args:
        img: Input image (T1 or FLAIR)
        cfg: Configuration object
        lg: Logger
        
    Returns:
        Tuple of (brain_image, brain_mask)
    """
    # Initialize variables
    brain, mask = None, None
    success = False
    
    # Try ANTs if in skullstrip_order
    if "ants" in cfg.skullstrip_order:
        try:
            lg.info("Skull-strip (ANTs)")
            mask = ants.brain_extraction(
                img, template=cfg.brain_template, brainmask=cfg.brain_mask_tpl, verbose=False
            )[1]
            
            # Check if ANTs result is satisfactory
            vox_count = mask.numpy().sum()
            if vox_count < 5000:
                lg.warning("ANTs skull-stripping produced small mask (%d voxels)", vox_count)
                success = False
            else:
                lg.info("ANTs skull-stripping successful")
                success = True
                brain = img * mask
        except Exception as e:
            lg.warning("ANTs skull-stripping failed: %s", e)
            success = False
    
    # Try SynthStrip if ANTs failed or not in skullstrip_order
    if (not success and "synthstrip" in cfg.skullstrip_order and SYNTHSTRIP_AVAILABLE):
        try:
            lg.info("Skull-strip (SynthStrip)")
            
            # Initialize SynthStrip
            stripper = SynthStrip()
            
            # Get mask using SynthStrip (returns numpy array)
            mask_array = stripper.run(img.numpy(), img.spacing)
            
            # Convert mask back to ANTs format
            mask = ants.from_numpy(
                mask_array.astype(np.uint8),
                origin=img.origin,
                spacing=img.spacing,
                direction=img.direction
            )
            
            # Apply mask to get brain
            brain = img * mask
            
            lg.info("SynthStrip skull-stripping successful")
            success = True
        except Exception as e:
            lg.warning("SynthStrip skull-stripping failed: %s", e)
            success = False
    
    # If all methods failed, try basic thresholding as last resort
    if not success:
        lg.warning("All skull-stripping methods failed - using basic thresholding")
        # Simple intensity-based threshold
        arr = img.numpy()
        threshold = np.percentile(arr[arr > 0], 25)  # Use 25th percentile of non-zero voxels
        mask_array = (arr > threshold).astype(np.uint8)
        
        # Create mask
        mask = ants.from_numpy(
            mask_array,
            origin=img.origin,
            spacing=img.spacing,
            direction=img.direction
        )
        
        # Apply mask to get brain
        brain = img * mask
    
    return brain, mask


# -----------------------------------------------------------------------------#
#  Pre-processing pipelines
# -----------------------------------------------------------------------------#

def preprocess_flair(flair:"ants.ANTsImage",
                     cfg:  Config,
                     lg )->"ants.ANTsImage":
    """Preprocess FLAIR image with denoising, N4 bias correction, and histogram matching.
    
    Args:
        flair: Input FLAIR image
        cfg: Configuration object
        lg: Logger instance
        
    Returns:
        Preprocessed FLAIR image
    """
    lg.info("FLAIR: denoise → N4 → histogram-match")
    den = ants.denoise_image(flair, noise_model="Rician", verbose=False)
    n4  = ants.n4_bias_field_correction(den, shrink_factor=cfg.flair_n4_shrink, verbose=False)
    tpl = ants.image_read(cfg.flair_template)
    match = ants.histogram_match_image(n4, tpl,
                                       number_of_histogram_levels=256,
                                       number_of_match_points=15)
    return match.astype("float32")


def preprocess_generic(mod:"ants.ANTsImage",
                       lg )->"ants.ANTsImage":
    """Apply generic preprocessing for non-FLAIR modalities (denoising + N4).
    
    Args:
        mod: Input image
        lg: Logger instance
        
    Returns:
        Preprocessed image
    """
    lg.debug("Generic preproc: denoise+N4")
    den = ants.denoise_image(mod, noise_model="Gaussian", verbose=False)
    return ants.n4_bias_field_correction(den, shrink_factor=1, verbose=False).astype("float32")


# -----------------------------------------------------------------------------#
#  Registration helpers
# -----------------------------------------------------------------------------#

def _rigid_bbr(fixed:"ants.ANTsImage",
               moving:"ants.ANTsImage")->Dict:
    """Perform rigid registration with boundary-based refinement.
    
    Creates an edge mask using the moving image gradient to improve registration.
    
    Args:
        fixed: Fixed/reference image
        moving: Moving image to register
        
    Returns:
        ANTs registration result dictionary
    """
    grad = ants.iMath(moving, "Grad", 1)
    edge = ants.threshold_image(grad, np.percentile(grad.numpy(),75), None, 1, 0)
    return ants.registration(fixed=fixed, moving=moving,
                             type_of_transform="Rigid",
                             aff_metric="MI", mask=edge,
                             verbose=False)

def register_to_t1(t1:"ants.ANTsImage",
                   mov:"ants.ANTsImage",
                   cfg:Config,
                   lg,
                   label:str)->Tuple["ants.ANTsImage",List[str],List[str]]:
    """Rigid(BBR) + SyN registration of input image to T1.
    
    Performs a two-stage registration:
    1. Rigid alignment using boundary-based registration
    2. Deformable SyN registration for fine alignment
    
    Args:
        t1: T1 brain image (target)
        mov: Moving image to register (source)
        cfg: Configuration object
        lg: Logger instance
        label: Label for the modality being registered
        
    Returns:
        Tuple of (warped_image, forward_transforms, inverse_transforms)
    """
    lg.info("Register %s → T1", label)
    rig = _rigid_bbr(t1, mov)
    mov_r = rig["warpedmovout"]
    syn   = ants.registration(fixed=t1, moving=mov_r,
                              type_of_transform="SyN",
                              syn_metric=cfg.reg_metric,
                              verbose=False)
    warped = syn["warpedmovout"]
    fwd    = rig["fwdtransforms"] + syn["fwdtransforms"]
    inv    = syn["invtransforms"] + rig["invtransforms"]
    return warped, fwd, inv


# -----------------------------------------------------------------------------#
#  Atlas masks
# -----------------------------------------------------------------------------#

def brainstem_masks(t1:"ants.ANTsImage",
                    inv:List[str],
                    cfg:Config,
                    lg )->Tuple["ants.ANTsImage","ants.ANTsImage","ants.ANTsImage"]:
    """Generate brainstem masks using atlas in subject space.
    
    Creates three masks:
    1. Full brainstem mask
    2. Dorsal mask (simple z-coordinate split)
    3. Ventral mask (simple z-coordinate split)
    
    Args:
        t1: T1 brain image
        inv: Inverse transforms to apply to atlas
        cfg: Configuration object
        lg: Logger instance
        
    Returns:
        Tuple of (full_brainstem_mask, dorsal_mask, ventral_mask)
    """
    if not Path(cfg.bs_mask_tpl).exists():
        lg.warning("Brain-stem atlas missing → zero masks")
        shp=t1.shape
        zero=ants.from_numpy(np.zeros(shp,np.uint8),origin=t1.origin,
                             spacing=t1.spacing,direction=t1.direction)
        return zero,zero,zero

    bs_tpl = ants.image_read(cfg.bs_mask_tpl)
    bs_sub = ants.apply_transforms(fixed=t1, moving=bs_tpl,
                                   transformlist=inv,
                                   interpolator="nearestNeighbor")
    bs_sub = ants.threshold_image(bs_sub,0.5,1.0,1,0).astype("uint8")
    # dorsal/ventral split
    zmin,zmax=np.where(bs_sub.numpy())[2].min(), np.where(bs_sub.numpy())[2].max()
    zmid=int((zmin+zmax)/2)
    dorsal = bs_sub*0; dorsal.numpy()[:,:,zmid:]=bs_sub.numpy()[:,:,zmid:]
    ventral= bs_sub*0; ventral.numpy()[:,:,:zmid]=bs_sub.numpy()[:,:,:zmid]
    return bs_sub,dorsal,ventral


# -----------------------------------------------------------------------------#
#  WM segmentation & lesion-detection (unchanged functions from v1.1)
# -----------------------------------------------------------------------------#

def wm_segmentation(t1_brain:"ants.ANTsImage",
                    flair_reg:"ants.ANTsImage",
                    brain_mask:"ants.ANTsImage",
                    lg) -> Tuple["ants.ANTsImage", Optional["ants.ANTsImage"]]:
    """Segment white matter using multi-modal information.
    
    Uses ANTs Atropos segmentation with T1 and FLAIR when T1 is available,
    otherwise falls back to simple Otsu thresholding on FLAIR.
    
    Args:
        t1_brain: T1 brain image (optional, can be None)
        flair_reg: FLAIR image registered to T1 space
        brain_mask: Brain mask
        lg: Logger instance
        
    Returns:
        Tuple of (white_matter_mask, lesion_mask)
        Note: lesion_mask may be None depending on segmentation method
    """
    if t1_brain:
        lg.info("Atropos WM segmentation")
        seg = ants.atropos(a=[t1_brain, flair_reg], x=brain_mask,
                           i="KMeans[4]", m="[0.2,1x1x1]", verbose=False)["segmentation_image"]
        wm  = seg==3; les=seg==4
    else:
        lg.info("Fallback Otsu WM")
        arr=flair_reg.numpy(); thr=np.percentile(arr,[33,66]); lab=np.digitize(arr,thr)
        wm = ants.from_numpy((lab==2).astype(np.uint8), origin=flair_reg.origin,
                             spacing=flair_reg.spacing, direction=flair_reg.direction)
        les=None
    return wm,les


def _mu_sigma(vals):
    """Calculate robust mean and standard deviation using median and MAD.
    
    Args:
        vals: Array of values
        
    Returns:
        Tuple of (robust_mean, robust_std)
    """
    med=np.median(vals)
    mad=np.median(np.abs(vals-med))
    return med,1.4826*mad

def lesion_masks(flair:"ants.ANTsImage",
                 wm:"ants.ANTsImage",
                 bs:"ants.ANTsImage",
                 cfg:Config,
                 lg) -> Dict[float, "ants.ANTsImage"]:
    """Generate lesion masks at multiple threshold levels.
    
    Uses adaptive thresholding based on normal-appearing white matter statistics.
    Applies morphological opening to remove small noise components.
    
    Args:
        flair: FLAIR image
        wm: White matter mask
        bs: Brainstem mask
        cfg: Configuration object
        lg: Logger instance
        
    Returns:
        Dictionary of lesion masks keyed by SD threshold
    """
    wm_vals=flair.numpy()[wm.numpy().astype(bool)]
    mu,sd=_mu_sigma(wm_vals)
    lg.debug("WM µ %.2f σ %.2f",mu,sd)
    struct=generate_binary_structure(3,3 if cfg.connectivity==26 else 1)
    masks={}
    for s in cfg.thresholds_sd:
        thr=mu+s*sd; lg.info("Threshold %.1f SD → %.2f",s,thr)
        vox=(flair.numpy()>thr)&(bs.numpy()>0)
        vox=binary_opening(vox,structure=struct,iterations=cfg.open_radius)
        lbl=label(vox,connectivity=cfg.connectivity)
        arr=np.zeros_like(vox,np.uint8)
        for ridx in range(1,lbl.max()+1):
            if (lbl==ridx).sum()>=cfg.min_vox: arr[lbl==ridx]=1
        masks[s]=ants.from_numpy(arr,origin=flair.origin,spacing=flair.spacing,direction=flair.direction)
    return masks


def cluster_metrics(mask:"ants.ANTsImage",
                    cfg:Config,
                    out:Path,
                    lg)->Path:
    """Calculate metrics for each lesion cluster and save to CSV.
    
    Computes size, shape, and location metrics for each cluster.
    Filters out clusters with excessive elongation.
    
    Args:
        mask: Binary lesion mask
        cfg: Configuration object
        out: Output directory
        lg: Logger instance
        
    Returns:
        Path to the generated CSV file
    """
    arr=mask.numpy().astype(bool); lbl=label(arr,connectivity=cfg.connectivity)
    if lbl.max()==0:
        csv=out/"lesion_clusters.csv"; pd.DataFrame().to_csv(csv,index=False); return csv
    props=regionprops_table(lbl,properties=("label","area","bbox","centroid"))
    df=pd.DataFrame(props); df["mm3"]=df["area"]*np.prod(mask.spacing)
    dims=np.stack([df[f"bbox-{i+3}"]-df[f"bbox-{i}"] for i in range(3)],1)
    df["elong"]=dims.max(1)/np.maximum(dims.min(1),1)
    df=df[df["elong"]<=cfg.max_elongation].sort_values("mm3",ascending=False)
    csv=out/"lesion_clusters.csv"; df.to_csv(csv,index=False)
    lg.info("Cluster CSV rows=%d", df.shape[0])
    return csv


def qc_overlay(base:"ants.ANTsImage",
               ov  :"ants.ANTsImage",
               fp  :Path,
               title:str):
    """Create quality control overlay image.
    
    Generates a 2D slice visualization with base image in grayscale
    and overlay image in color.
    
    Args:
        base: Base image (typically FLAIR)
        ov: Overlay image (typically lesion mask)
        fp: Output file path
        title: Title for the plot
    """
    try:
        z=base.shape[2]//2
        fig,ax=plt.subplots(figsize=(6,6))
        ax.imshow(base.numpy()[:,:,z].T,cmap="gray",origin="lower")
        ax.imshow(np.ma.masked_where(ov.numpy()[:,:,z]==0,ov.numpy()[:,:,z]).T,
                  cmap="autumn",alpha=0.5,origin="lower")
        ax.set_title(title); ax.axis("off")
        fig.savefig(fp,dpi=140,bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        logging.getLogger(__name__).warning("QC overlay failed: %s", e)
