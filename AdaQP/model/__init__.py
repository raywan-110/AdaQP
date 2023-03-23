from enum import Enum
from .distGCN import distGCN
from .distSAGE import distSAGE

class DistGNNType(Enum):
    DistGCN = 0
    DistGraphSAGE = 1