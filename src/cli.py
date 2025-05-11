from __future__ import annotations
import argparse, sys
from pathlib import Path
from .core import Config
from .pipeline import process_subject, Parallel, process_batch

def main():
    ap = argparse.ArgumentParser(
        prog="brainstemx",
        description="BrainStem X - Brain-Stem / Pons MRI Intensity-Clustering Pipeline"
    )
    # Required arguments
    ap.add_argument("--flair", required=True, help="FLAIR NIfTI image (required)")
    ap.add_argument("--out", required=True, help="Output directory")
    
    # Optional arguments
    ap.add_argument("--t1", help="T1 NIfTI image (optional but recommended)")
    ap.add_argument("--dwi", help="DWI NIfTI image (optional)")
    ap.add_argument("--swi", help="SWI NIfTI image (optional)")
    ap.add_argument("--list", help="Batch file with subject details")
    ap.add_argument("--cfg", help="Configuration JSON file")
    ap.add_argument("--jobs", type=int, help="Number of parallel jobs")
    
    a = ap.parse_args()
    
    try:
        # Load configuration
        cfg = Config.load(Path(a.cfg)) if a.cfg else Config()
        if a.jobs: cfg.n_jobs = a.jobs
        cfg.resolve()
        
        if a.list:
            # Batch processing
            if not Path(a.list).exists():
                raise FileNotFoundError(f"Batch list file not found: {a.list}")
            process_batch(Path(a.list), cfg)
        else:
            # Single subject processing
            # Check required and optional inputs
            if not Path(a.flair).exists():
                raise FileNotFoundError(f"FLAIR file not found: {a.flair}")
            if a.t1 and not Path(a.t1).exists():
                raise FileNotFoundError(f"T1 file not found: {a.t1}")
            if a.dwi and not Path(a.dwi).exists():
                raise FileNotFoundError(f"DWI file not found: {a.dwi}")
            if a.swi and not Path(a.swi).exists():
                raise FileNotFoundError(f"SWI file not found: {a.swi}")
            
            # Create output directory if it doesn't exist
            out_path = Path(a.out)
            out_path.mkdir(parents=True, exist_ok=True)
            
            # Pass all files to pipeline including optional modalities
            process_subject(
                out_path.stem,
                Path(a.flair),
                Path(a.t1) if a.t1 else None,
                out_path,
                cfg,
                Path(a.dwi) if a.dwi else None,
                Path(a.swi) if a.swi else None
            )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())

