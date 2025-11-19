import os
import random
import time
from contextlib import contextmanager
import numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

@contextmanager
def timer(name: str):
    start = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - start:.1f}s")
