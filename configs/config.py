from pathlib import Path


class Config:
    # ── Paths ──────────────────────────────────────────────
    DATA_ROOT  = Path("data/raw/brats2020")
    TRAIN_DIR  = DATA_ROOT / "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    OUTPUT_DIR = Path("results")
    CHECKPOINT = OUTPUT_DIR / "best_model.pth"

    # ── MRI Modalities ────────────────────────────────────
    MODALITIES   = ["flair", "t1", "t1ce", "t2"]
    IN_CHANNELS  = 4
    OUT_CHANNELS = 3  # WT, TC, ET

    # ── Spatial ───────────────────────────────────────────
    ROI_SIZE = (96, 96, 96)
    PIXDIM   = (1.5, 1.5, 1.5)

    # ── Training ──────────────────────────────────────────
    EPOCHS        = 50
    BATCH_SIZE    = 1
    VAL_FREQ      = 2
    LR            = 2e-4
    WEIGHT_DECAY  = 1e-5
    WARMUP_EPOCHS = 5
    GRAD_CLIP     = 1.0
    AMP           = True
    NUM_WORKERS   = 2
    CACHE_RATE    = 0.25

    # ── Model ─────────────────────────────────────────────
    FEATURE_SIZE   = 48
    DROPOUT        = 0.1
    ATTN_DROP      = 0.0
    USE_CHECKPOINT = True

    # ── Uncertainty ───────────────────────────────────────
    MC_PASSES = 10

    # ── Misc ──────────────────────────────────────────────
    SEED      = 42
    VAL_SPLIT = 0.15

    # ── Labels ────────────────────────────────────────────
    LABEL_NAMES  = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
    LABEL_COLORS = ["#4CAF50", "#FF9800", "#F44336"]


cfg = Config()
