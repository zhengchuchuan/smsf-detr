import os                                                                                                                                                       
import random                                                                                                                                                   
import shutil                                                                                                                                                   
from pathlib import Path                                                                                                                                        
from typing import Dict, Iterable, List, Sequence, Tuple                                                                                                        
                                                                                                                                                                
                                                                                                                                                                
def _ensure_dir(p: Path) -> None:                                                                                                                               
    p.mkdir(parents=True, exist_ok=True)                                                                                                                        
                                                                                                                                                                
                                                                                                                                                                
def _collect_by_basename(root: Path, exts: Sequence[str]) -> Dict[str, Path]:                                                                                   
    """                                                                                                                                                         
    Scan a folder and return a mapping: basename -> full path                                                                                                   
    Only the first file encountered for a basename wins (deterministic by sort).                                                                                
    """                                                                                                                                                         
    exts = tuple(e.lower() for e in exts)                                                                                                                       
    mapping: Dict[str, Path] = {}                                                                                                                               
    if not root.exists():                                                                                                                                       
        return mapping                                                                                                                                          
    for p in sorted(root.iterdir()):                                                                                                                            
        if not p.is_file():                                                                                                                                     
            continue                                                                                                                                            
        ext = p.suffix.lower()                                                                                                                                  
        if ext in exts:                                                                                                                                         
            base = p.stem                                                                                                                                       
            if base not in mapping:                                                                                                                             
                mapping[base] = p                                                                                                                               
    return mapping                                                                                                                                              
                                                                                                                                                                
                                                                                                                                                                
def _copy(src: Path, dst: Path, overwrite: bool = False) -> None:                                                                                               
    if dst.exists() and not overwrite:                                                                                                                          
        return                                                                                                                                                  
    _ensure_dir(dst.parent)                                                                                                                                     
    shutil.copy2(src, dst)                                                                                                                                      
                                                                                                                                                                
                                                                                                                                                                
def _split_indices(n: int, split_ratio: Tuple[float, float, float]) -> Tuple[List[int], List[int], List[int]]:                                                  
    # compute counts with remainder assigned to test to keep sum == n                                                                                           
    train_n = int(n * split_ratio[0])                                                                                                                           
    val_n = int(n * split_ratio[1])                                                                                                                             
    test_n = n - train_n - val_n                                                                                                                                
    idxs = list(range(n))                                                                                                                                       
    return idxs[:train_n], idxs[train_n:train_n + val_n], idxs[train_n + val_n:]                                                                                
                                                                                                                                                                
                                                                                                                                                                
def split_dataset_rgb_spectral(                                                                                                                                 
    rgb_dir: str,                                                                                                                                               
    spectral_dir: str,                                                                                                                                          
    labels_dir: str,                                                                                                                                            
    output_dir: str,                                                                                                                                            
    *,                                                                                                                                                          
    label_suffix: str = ".json",                                                                                                                                
    spectral_ext: str = ".tif",                                                                                                                                 
    rgb_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),                                                                                                        
    split_ratio: Tuple[float, float, float] = (0.8, 0.2, 0.0),                                                                                                  
    seed: int | None = 42,                                                                                                                                      
    labels_subdir_name: str = "xanylabeling",                                                                                                                   
    strict: bool = True,                                                                                                                                        
    overwrite: bool = False,                                                                                                                                    
    dry_run: bool = False,                                                                                                                                      
) -> None:                                                                                                                                                      
    """                                                                                                                                                         
    Create train/val/test splits by copying matched triples:                                                                                                    
    - RGB image from rgb_dir                                                                                                                                  
    - Spectral image (.tif) from spectral_dir                                                                                                                 
    - Label file from labels_dir                                                                                                                              
                                                                                                                                                                
    Pairing is by basename (e.g., foo.jpg <-> foo.tif <-> foo.json).                                                                                            
    When strict=True, only samples that have all three files are included.                                                                                      
                                                                                                                                                                
    Output layout (example for train):                                                                                                                          
    output_dir/train/rgb                                                                                                                                      
    output_dir/train/spectral                                                                                                                                 
    output_dir/train/<labels_subdir_name>                                                                                                                     
    """                                                                                                                                                         
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "split_ratio must sum to 1.0"                                                                                    
                                                                                                                                                                
    if seed is not None:                                                                                                                                        
        random.seed(seed)                                                                                                                                       
                                                                                                                                                                
    rgb_root = Path(rgb_dir)                                                                                                                                    
    sp_root = Path(spectral_dir)                                                                                                                                
    lb_root = Path(labels_dir)                                                                                                                                  
    out_root = Path(output_dir)                                                                                                                                 
                                                                                                                                                                
    # Collect available files                                                                                                                                   
    rgb_map = _collect_by_basename(rgb_root, rgb_exts)                                                                                                          
    sp_map = _collect_by_basename(sp_root, (spectral_ext,))                                                                                                     
    lb_map = _collect_by_basename(lb_root, (label_suffix,))                                                                                                     
                                                                                                                                                                
    rgb_bases = set(rgb_map.keys())                                                                                                                             
    sp_bases = set(sp_map.keys())                                                                                                                               
    lb_bases = set(lb_map.keys())                                                                                                                               
                                                                                                                                                                
    if strict:                                                                                                                                                  
        bases = sorted(rgb_bases & sp_bases & lb_bases)                                                                                                         
    else:                                                                                                                                                       
        # keep samples that have at least RGB + label; spectral optional                                                                                        
        bases = sorted((rgb_bases & lb_bases))                                                                                                                  
                                                                                                                                                                
    # Report missing pairs for visibility                                                                                                                       
    missing_spectral = sorted((rgb_bases & lb_bases) - sp_bases)                                                                                                
    missing_labels = sorted(rgb_bases - lb_bases)                                                                                                               
    missing_rgb = sorted((lb_bases & sp_bases) - rgb_bases)                                                                                                     
                                                                                                                                                                
    print(f"Found: RGB={len(rgb_bases)}, spectral={len(sp_bases)}, labels={len(lb_bases)}")                                                                     
    if missing_spectral:                                                                                                                                        
        print(f"Missing spectral for {len(missing_spectral)} samples (by basename).")                                                                           
    if missing_labels:                                                                                                                                          
        print(f"Missing labels for {len(missing_labels)} samples (by basename).")                                                                               
    if missing_rgb:                                                                                                                                             
        print(f"Missing RGB for {len(missing_rgb)} samples (by basename).")                                                                                     
                                                                                                                                                                
    # Build triples (or pairs if not strict)                                                                                                                    
    samples: List[Tuple[str, Path, Path | None, Path]] = []                                                                                                     
    for b in bases:                                                                                                                                             
        rgb_p = rgb_map[b]                                                                                                                                      
        sp_p = sp_map.get(b)                                                                                                                                    
        lb_p = lb_map.get(b)                                                                                                                                    
        if strict:                                                                                                                                              
            # must have all                                                                                                                                     
            if rgb_p is None or sp_p is None or lb_p is None:                                                                                                   
                continue                                                                                                                                        
        else:                                                                                                                                                   
            # RGB+label required, spectral optional                                                                                                             
            if rgb_p is None or lb_p is None:                                                                                                                   
                continue                                                                                                                                        
        samples.append((b, rgb_p, sp_p, lb_p))  # base, rgb, spectral?, label                                                                                   
                                                                                                                                                                
    n = len(samples)
    if n == 0:                                                                                                                                                  
        print("No valid samples to split. Nothing to do.")                                                                                                      
        return                                                                                                                                                  
                                                                                                                                                                
    # Shuffle by basename for stable randomization                                                                                                              
    random.shuffle(samples)                                                                                                                                     
                                                                                                                                                                
    train_idx, val_idx, test_idx = _split_indices(n, split_ratio)                                                                                               
    splits = {                                                                                                                                                  
        "train": [samples[i] for i in train_idx],                                                                                                               
        "val": [samples[i] for i in val_idx],                                                                                                                   
        "test": [samples[i] for i in test_idx],                                                                                                                 
    }                                                                                                                                                           
                                                                                                                                                                
    # Prepare destination subdir names                                                                                                                          
    sub_rgb = "rgb"                                                                                                                                             
    sub_sp = "spectral"                                                                                                                                         
    sub_lb = labels_subdir_name                                                                                                                                 
                                                                                                                                                                
    # Copy                                                                                                                                                      
    for split_name, split_samples in splits.items():                                                                                                            
        if not split_samples:                                                                                                                                   
            continue                                                                                                                                            
        dest_rgb = out_root / split_name / sub_rgb                                                                                                              
        dest_sp = out_root / split_name / sub_sp                                                                                                                
        dest_lb = out_root / split_name / sub_lb                                                                                                                
                                                                                                                                                                
        if dry_run:                                                                                                                                             
            print(f"[dry-run] Would copy {len(split_samples)} samples to {split_name}/{{{sub_rgb},{sub_sp},{sub_lb}}}")                                         
            continue                                                                                                                                            
                                                                                                                                                                
        _ensure_dir(dest_rgb)                                                                                                                                   
        _ensure_dir(dest_sp)                                                                                                                                    
        _ensure_dir(dest_lb)                                                                                                                                    
                                                                                                                                                                
        for base, rgb_p, sp_p, lb_p in split_samples:                                                                                                           
            # Keep original filenames and extensions                                                                                                            
            _copy(rgb_p, dest_rgb / rgb_p.name, overwrite=overwrite)                                                                                            
            if sp_p is not None:                                                                                                                                
                _copy(sp_p, dest_sp / sp_p.name, overwrite=overwrite)                                                                                           
            if lb_p is not None:                                                                                                                                
                _copy(lb_p, dest_lb / lb_p.name, overwrite=overwrite)                                                                                           
                                                                                                                                                                
    print("Done.")                                                                                                                                              
    print(f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")                                                             
                                                                                                                                                                
                                                                                                                                                                
if __name__ == "__main__":                                                                                                                                      
    import argparse                                                                                                                                             
                                                                                                                                                                
    parser = argparse.ArgumentParser(                                                                                                                           
        description="Split dataset into train/val/test and copy rgb, spectral(.tif), and labels into separate folders."                                         
    )                                                                                                                                                           
    parser.add_argument("--rgb-dir", required=True, help="Folder with RGB images (e.g., .jpg/.png)")
    parser.add_argument("--spectral-dir", required=True, help="Folder with spectral images (.tif)")                                                             
    parser.add_argument("--labels-dir", required=True, help="Folder with annotation files (e.g., .json or .txt)")                                               
    parser.add_argument("--output-dir", required=True, help="Output dataset root")                                                                              

    parser.add_argument("--label-suffix", default=".json", help="Label file extension/suffix (default: .json)")                                                 
    parser.add_argument("--spectral-ext", default=".tif", help="Spectral image extension (default: .tif)")                                                      
    parser.add_argument(                                                                                                                                        
        "--rgb-exts",                                                                                                                                           
        nargs="+",                                                                                                                                              
        default=[".jpg", ".jpeg", ".png"],                                                                                                                      
        help="Allowed RGB extensions (space-separated). Default: .jpg .jpeg .png",                                                                              
    )                                                                                                                                                           
    parser.add_argument(                                                                                                                                        
        "--split-ratio",                                                                                                                                        
        nargs=3,                                                                                                                                                
        type=float,                                                                                                                                             
        default=[0.8, 0.2, 0.0],                                                                                                                                
        metavar=("TRAIN", "VAL", "TEST"),                                                                                                                       
        help="Three floats summing to 1.0 (default: 0.8 0.2 0.0)",                                                                                              
    )                                                                                                                                                           
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")                                                                       
    parser.add_argument("--labels-subdir-name", default="xanylabeling", help="Subfolder name for labels under each split")                                      
    parser.add_argument(                                                                                                                                        
        "--allow-missing",                                                                                                                                      
        action="store_true",                                                                                                                                    
        help="Do not require spectral for every RGB+label; copy RGB+label anyway (spectral optional)",                                                          
    )                                                                                                                                                           
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files at destination")                                                     
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done; no files copied")                                                
                                                                                                                                                                
    args = parser.parse_args()                                                                                                                                  
    split_dataset_rgb_spectral(                                                                                                                                 
        rgb_dir=args.rgb_dir,                                                                                                                                   
        spectral_dir=args.spectral_dir,                                                                                                                         
        labels_dir=args.labels_dir,                                                                                                                             
        output_dir=args.output_dir,                                                                                                                             
        label_suffix=args.label_suffix,                                                                                                                         
        spectral_ext=args.spectral_ext,                                                                                                                         
        rgb_exts=tuple(args.rgb_exts),                                                                                                                          
        split_ratio=tuple(args.split_ratio),  # type: ignore[arg-type]                                                                                          
        seed=args.seed,                                                                                                                                         
        labels_subdir_name=args.labels_subdir_name,                                                                                                             
        strict=not args.allow_missing,                                                                                                                          
        overwrite=args.overwrite,                                                                                                                               
        dry_run=args.dry_run,                                                                                                                                   
    ) 

"""
python utils/data_process/split_dataset_rgb_spectral.py \
--rgb-dir "/mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/clip/rgb"     \
--spectral-dir "/mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/clip/spectral" \
--labels-dir "/mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/clip/xanylabeling" \
--output-dir "/mnt/d/Project/master-graduation-project/data/oil/oil_dataset_20260109_简单目标/clip/dataset"
"""
