from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Input
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.NAME = 'C3D.train'
_C.TRAIN.MAX_ITER = 10000
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_STEP = 2500
_C.TRAIN.EVAL_STEP = 2500

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.INFER = CN()
_C.INFER.NAME = 'C3D.infer'

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'c3d'
# HxWxC
_C.MODEL.INPUT_SIZE = (112, 112, 3)
_C.MODEL.NUM_CLASSES = 51

# ---------------------------------------------------------------------------- #
# Criterion
# ---------------------------------------------------------------------------- #
_C.CRITERION = CN()
_C.CRITERION.NAME = 'crossentropy'

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'sgd'
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 3e-4
# for sgd
_C.OPTIMIZER.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# LR_Scheduler
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = 'multistep_lr'
# for SteLR
_C.LR_SCHEDULER.STEP_SIZE = 400
# for MultiStepLR
_C.LR_SCHEDULER.MILESTONES = [2500, 6000]

# ---------------------------------------------------------------------------- #
# DataSets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()

_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.NAME = 'HMDB51'
_C.DATASETS.TRAIN.STEP_BETWEEN_CLIPS = 16
_C.DATASETS.TRAIN.FRAMES_PER_CLIP = 16
_C.DATASETS.TRAIN.VIDEO_DIR = 'data/hmdb51_org'
_C.DATASETS.TRAIN.ANNOTATION_DIR = 'data/testTrainMulti_7030_splits'

_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.NAME = 'HMDB51'
_C.DATASETS.TEST.STEP_BETWEEN_CLIPS = 16
_C.DATASETS.TEST.FRAMES_PER_CLIP = 16
_C.DATASETS.TEST.VIDEO_DIR = 'data/hmdb51_org'
_C.DATASETS.TEST.ANNOTATION_DIR = 'data/testTrainMulti_7030_splits'

# ---------------------------------------------------------------------------- #
# Transform
# ---------------------------------------------------------------------------- #
_C.TRANSFORM = CN()
_C.TRANSFORM.MEAN = (0.5, 0.5, 0.5)
_C.TRANSFORM.STD = (0.5, 0.5, 0.5)

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 16
_C.DATALOADER.TEST_BATCH_SIZE = 16
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = 'outputs/hmdb51'
