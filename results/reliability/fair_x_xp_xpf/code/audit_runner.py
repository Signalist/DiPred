"""Run the frozen audit with JSON serialization for NumPy scalar booleans."""
import json
import runpy
from pathlib import Path
import numpy as np
original=json.dumps

def dumps(value,*args,**kwargs):
    kwargs.setdefault('default',lambda item: item.item() if isinstance(item,np.generic) else str(item))
    return original(value,*args,**kwargs)

json.dumps=dumps
runpy.run_path(str(Path(__file__).with_name('audit.py')),run_name='__main__')
