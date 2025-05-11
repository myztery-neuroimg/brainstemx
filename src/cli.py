from __future__ import annotations
import argparse
from pathlib import Path
from .core import Config
from .pipeline import process_subject, Parallel, process_batch

def main():
    ap=argparse.ArgumentParser("brainstemx")
    ap.add_argument("--flair"); ap.add_argument("--t1"); ap.add_argument("--out")
    ap.add_argument("--list");  ap.add_argument("--cfg"); ap.add_argument("--jobs", type=int)
    a=ap.parse_args()
    cfg=Config.load(Path(a.cfg)) if a.cfg else Config()
    if a.jobs: cfg.n_jobs=a.jobs
    cfg.resolve()
    if a.list: process_batch(Path(a.list),cfg)
    else:
        process_subject(Path(a.out).stem, Path(a.flair),
                        Path(a.t1) if a.t1 else None, Path(a.out), cfg)

if __name__=="__main__": main()

