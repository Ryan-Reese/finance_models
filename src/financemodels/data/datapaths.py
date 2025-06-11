from pathlib import Path
from enum import Enum


class DeepLOBDataPaths(Enum):

    DIR_PATH = Path("data/fi-2010")
    TRAIN_PATH = DIR_PATH.glob("Train*")
    VAL_PATH = DIR_PATH.glob("Train*")
    TEST_PATH = DIR_PATH.glob("Test*")
