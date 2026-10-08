# Ultralytics YOLO, AGPL-3.0 license
__version__ = "8.3.50"
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from ultralytics.models import YOLOFT
from ultralytics.utils import ASSETS, SETTINGS
settings = SETTINGS
__all__ = ("YOLOFT", "ASSETS", "settings", "__version__")
