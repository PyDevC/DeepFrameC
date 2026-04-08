import os

class Config:
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "FaceForensics")
    MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    COMPRESSION = "c23"
    FRAMES_PER_VIDEO = 10
    FRAMES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "FaceForensics_transformed")
    FACE_SIZE = 22

    TRAIN_SPLIT = "train"
    VAL_SPLIT   = "val"
    TEST_SPLIT  = "test"

    BATCH_SIZE  = 32
    NUM_WORKERS = 8
    EPOCHS      = 5
    LR          = 1e-4
    WEIGHT_DECAY = 1e-5
    LABEL_SMOOTHING = 0.1

    BACKBONE    = "efficientnet_b4"
    PRETRAINED  = True
    DROPOUT     = 0.5
    NUM_CLASSES = 2

    CHECKPOINT_DIR = "checkpoints/"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
