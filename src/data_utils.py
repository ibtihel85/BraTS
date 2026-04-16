import random
from pathlib import Path
from typing import List, Dict

import nibabel as nib
import numpy as np
from tqdm import tqdm

from configs.config import cfg


def discover_cases(root_dir: Path) -> List[Dict]:
    cases = []
    patient_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    for pdir in patient_dirs:
        pid = pdir.name
        entry = {"pid": pid}
        for mod in cfg.MODALITIES:
            f = pdir / f"{pid}_{mod}.nii"
            if not f.exists():
                f = pdir / f"{pid}_{mod}.nii.gz"
            entry[mod] = str(f) if f.exists() else None
        seg_f = pdir / f"{pid}_seg.nii"
        if not seg_f.exists():
            seg_f = pdir / f"{pid}_seg.nii.gz"
        entry["seg"] = str(seg_f) if seg_f.exists() else None
        if all(entry[m] for m in cfg.MODALITIES) and entry["seg"]:
            cases.append(entry)
    return cases


def split_cases(all_cases: List[Dict], val_split: float = cfg.VAL_SPLIT, seed: int = cfg.SEED):
    random.seed(seed)
    cases = all_cases.copy()
    random.shuffle(cases)
    val_n = max(1, int(len(cases) * val_split))
    return cases[val_n:], cases[:val_n]


def make_datalist(cases: List[Dict]) -> List[Dict]:
    return [
        {**{mod: c[mod] for mod in cfg.MODALITIES}, "seg": c["seg"]}
        for c in cases
    ]


def analyze_label_distribution(cases: List[Dict], n_samples: int = 20) -> Dict:
    counts = {0: 0, 1: 0, 2: 0, 4: 0}
    for case in tqdm(cases[:n_samples], desc="Analyzing labels"):
        seg = nib.load(case["seg"]).get_fdata().astype(np.int32)
        for lbl in [0, 1, 2, 4]:
            counts[lbl] += int((seg == lbl).sum())
    return counts
