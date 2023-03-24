from enum import Enum

from .distGCN import DistGCN
from .distSAGE import DistSAGE

class DistGNNType(Enum):
    DistGCN = 0
    DistSAGE = 1