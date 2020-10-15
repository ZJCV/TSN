from yacs.config import CfgNode as CN

_C = CN()
_C.RNG_SEED = 0

_C.NUM_GPUS = 1
_C.NODES = 1
_C.RANK = 0
_C.WORLD_SIZE = 1

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.MAX_ITER = 30000
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_STEP = 1000
_C.TRAIN.EVAL_STEP = 1000
_C.TRAIN.RESUME = False
_C.TRAIN.USE_TENSORBOARD = True

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.INFER = CN()

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = 'outputs/'

# ---------------------------------------------------------------------------- #
# DataSets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TYPE = 'RawFrame'
_C.DATASETS.MODALITY = 'RGB'
_C.DATASETS.SAMPLE_STRATEGY = 'SegSample'
_C.DATASETS.CLIP_LEN = 1
_C.DATASETS.FRAME_INTERVAL = 1
_C.DATASETS.NUM_CLIPS = 3
# for vidoe decode
# Enable multi thread decoding.
_C.DATASETS.ENABLE_MULTI_THREAD_DECODE = False
# Decoding backend, options include `pyav` or `torchvision`
_C.DATASETS.DECODING_BACKEND = "pyav"
# train
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.NAME = 'HMDB51'
_C.DATASETS.TRAIN.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TRAIN.ANNOTATION_DIR = 'data/hmdb51'
# for hmdb51 and ucf101
_C.DATASETS.TRAIN.SPLIT = 1
# test
_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.NAME = 'HMDB51'
_C.DATASETS.TEST.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TEST.ANNOTATION_DIR = 'data/hmdb51'
# for hmdb51 and ucf101
_C.DATASETS.TEST.SPLIT = 1

# ---------------------------------------------------------------------------- #
# Transform
# ---------------------------------------------------------------------------- #
_C.TRANSFORM = CN()
_C.TRANSFORM.SCALE_JITTER = (256, 320)
_C.TRANSFORM.TRAIN_CROP_SIZE = 224
_C.TRANSFORM.TEST_CROP_SIZE = 256
_C.TRANSFORM.MEAN = (0.485, 0.456, 0.406)  # (0.5, 0.5, 0.5)
_C.TRANSFORM.STD = (0.229, 0.224, 0.225)  # (0.5, 0.5, 0.5)
_C.TRANSFORM.RANDOM_ROTATION = 10
# (brightness, contrast, saturation, hue)
_C.TRANSFORM.COLOR_JITTER = (0.1, 0.1, 0.1, 0.1)

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 16
_C.DATALOADER.TEST_BATCH_SIZE = 16
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = "TSN"
_C.MODEL.PRETRAINED = ""
_C.MODEL.SYNC_BN = False
_C.MODEL.INPUT_SIZE = (224, 224, 3)

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.BACKBONE.PARTIAL_BN = False
_C.MODEL.BACKBONE.TORCHVISION_PRETRAINED = False
_C.MODEL.BACKBONE.ZERO_INIT_RESIDUAL = False

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = 'TSNHead'
_C.MODEL.HEAD.FEATURE_DIMS = 2048
_C.MODEL.HEAD.DROPOUT = 0.0
_C.MODEL.HEAD.NUM_CLASSES = 51

_C.MODEL.RECOGNIZER = CN()
_C.MODEL.RECOGNIZER.NAME = 'TSNRecognizer'

_C.MODEL.CONSENSU = CN()
_C.MODEL.CONSENSU.NAME = 'AvgConsensus'

_C.MODEL.CRITERION = CN()
_C.MODEL.CRITERION.NAME = 'crossentropy'

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'sgd'
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 3e-5
# for sgd
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# LR_Scheduler
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = 'multistep_lr'
_C.LR_SCHEDULER.IS_WARMUP = False
_C.LR_SCHEDULER.GAMMA = 0.1

# for SteLR
_C.LR_SCHEDULER.STEP_LR = CN()
_C.LR_SCHEDULER.STEP_LR.STEP_SIZE = 10000
# for MultiStepLR
_C.LR_SCHEDULER.MULTISTEP_LR = CN()
_C.LR_SCHEDULER.MULTISTEP_LR.MILESTONES = [15000, 25000]
# for CosineAnnealingLR
_C.LR_SCHEDULER.COSINE_ANNEALING_LR = CN()
_C.LR_SCHEDULER.COSINE_ANNEALING_LR.MINIMAL_LR = 3e-5
# for Warmup
_C.LR_SCHEDULER.WARMUP = CN()
_C.LR_SCHEDULER.WARMUP.ITERATION = 400
_C.LR_SCHEDULER.WARMUP.MULTIPLIER = 1.0

# ---------------------------------------------------------------------------- #
# Demo options
# ---------------------------------------------------------------------------- #
_C.DEMO = CN()

# Run model in DEMO mode.
_C.DEMO.ENABLE = False

# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
_C.DEMO.LABEL_FILE_PATH = ""

# Specify a camera device as input. This will be prioritized
# over input manager if set.
# If -1, use input manager instead.
_C.DEMO.WEBCAM = -1

# Path to input manager for demo.
_C.DEMO.INPUT_VIDEO = ""
# Custom width for reading input manager data.
_C.DEMO.DISPLAY_WIDTH = 0
# Custom height for reading input manager data.
_C.DEMO.DISPLAY_HEIGHT = 0
# Number of overlapping frames between 2 consecutive clips.
# Increase this number for more frequent action predictions.
# The number of overlapping frames cannot be larger than
# half of the sequence length `cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE`
_C.DEMO.BUFFER_SIZE = 0
# If specified, the visualized outputs will be written this a manager file of
# this path. Otherwise, the visualized outputs will be displayed in a window.
_C.DEMO.OUTPUT_FILE = ""
# Frames per second rate for writing to output manager file.
# If not set (-1), use fps rate from input file.
_C.DEMO.OUTPUT_FPS = -1
# Input format from demo manager reader ("RGB" or "BGR").
_C.DEMO.INPUT_FORMAT = "BGR"
# Number of processes to run manager visualizer.
_C.DEMO.NUM_VIS_INSTANCES = 2

# Whether to run in with multi-threaded manager reader.
_C.DEMO.THREAD_ENABLE = False
# Take one clip for every `DEMO.NUM_CLIPS_SKIP` + 1 for prediction and visualization.
# This is used for fast demo speed by reducing the prediction/visualiztion frequency.
# If -1, take the most recent read clip for visualization. This mode is only supported
# if `DEMO.THREAD_ENABLE` is set to True.
_C.DEMO.NUM_CLIPS_SKIP = 0
# Visualize with top-k predictions or predictions above certain threshold(s).
# Option: {"thres", "top-k"}
_C.DEMO.VIS_MODE = "thres"
# Threshold for common class names.
_C.DEMO.COMMON_CLASS_THRES = 0.7
# Theshold for uncommon class names. This will not be
# used if `_C.DEMO.COMMON_CLASS_NAMES` is empty.
_C.DEMO.UNCOMMON_CLASS_THRES = 0.3
# This is chosen based on distribution of examples in
# each classes in AVA dataset.
_C.DEMO.COMMON_CLASS_NAMES = [
    "watch (a person)",
    "talk to (e.g., self, a person, a group)",
    "listen to (a person)",
    "touch (an object)",
    "carry/hold (an object)",
    "walk",
    "sit",
    "lie/sleep",
    "bend/bow (at the waist)",
]
# Slow-motion rate for the visualization. The visualized portions of the
# manager will be played `_C.DEMO.SLOWMO` times slower than usual speed.
_C.DEMO.SLOWMO = 1
# Colormap to for text boxes and bounding boxes colors
_C.DEMO.COLORMAP = "Pastel2"