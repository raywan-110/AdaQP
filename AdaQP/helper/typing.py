from enum import Enum

# GNN types
class DistGNNType(Enum):
    DistGCN = 0
    DistSAGE = 1

# BIT types
class BitType(Enum):
    '''
    bit width type.
    '''
    FULL = 0
    QUANT = 1

# Message types
class MessageType(Enum):
    '''
    message type for communication.
    '''
    DATA = 0
    PARAMs = 1

# propagation mode
class ProprogationMode(Enum):
    Forward = 0
    Backward = 1