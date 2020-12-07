from yacs.config import CfgNode as CN

_C = CN()

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# ---------------------------------------------------------------------------- #
# Distributed options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_NODES = 1

# The index of the current machine.
_C.RANK_ID = 0

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_EPOCH = 10
_C.TRAIN.EVAL_EPOCH = 10
_C.TRAIN.MAX_EPOCH = 100
_C.TRAIN.RESUME = False
_C.TRAIN.USE_TENSORBOARD = True