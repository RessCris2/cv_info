"""
  一些常用到的代码
"""
import os
import glob
from pathlib import Path

def get_id_list(file_dir):
    file_path = os.path.join(file_dir, "*.jpg")
    files = glob.glob(file_path)
    
    assert len(files) > 1
    ids = [int(Path(glob.glob(file)[0]).stem) for file in files]
    return ids