import os

class Config:
    DATA_ROOT  = "data/WaveFake"
    SAMPLE_RATE = 16000
    DURATION = 4 # seconds
    MAX_SAMPLES = SAMPLE_RATE * DURATION
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 512

    TRAIN_SPLIT = "train"
    VAL_SPLIT   = "val"
    TEST_SPLIT  = "test"

    BATCH_SIZE       = 32 
    GRAD_ACCUM_STEPS = 2
    NUM_WORKERS      = 8
    EPOCHS           = 30
    WARMUP_EPOCHS    = 5
    LR               = 1e-4
    WEIGHT_DECAY     = 1e-4

    PRETRAINED  = True
    DROPOUT     = 0.4
    NUM_CLASSES = 2
    CHECKPOINT_DIR = "checkpoints/"
