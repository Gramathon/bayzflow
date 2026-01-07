# bayzflow/__init__.py

__version__ = "0.1.0"

from bayzflow.core.engine import BayzFlow          # import the engine module
from bayzflow.core.experiment import exp as _exp    # import the exp() helper from experiment.py

# allow: BayzFlow.exp("my.yaml")
BayzFlow.exp = staticmethod(_exp)

__all__ = ["BayzFlow", "__version__"]
