from pathlib import Path
import os

class Environment(object):

  def __init__(self):
    pwd = Path(os.getcwd())
    if not os.path.isdir(pwd / 'data'):
      os.mkdir( pwd / 'data'   )
      os.mkdir( pwd / 'data' / 'source'   )
      os.mkdir( pwd / 'data' / 'models' )
      os.mkdir( pwd / 'data' / 'model_output' )

    self.frames = pwd / 'input_data' / 'frames'
    self.units = pwd / 'input_data' / 'units'
    
    self.output_data = pwd / 'output_data'

env = Environment()
