import torch
from typing import Dict, Tuple
from functools import wraps
from typing import Tuple
from torch import Tensor
import quant_cuda as integer_quantizer

from ..helper import BitType
from ..communicator import Basic_Buffer_Type
from ..communicator import Communicator as comm
from ..manager import GraphEngine as engine
from ..assigner import Assigner as assigner

'''
*************************************************
***** quantization/dequantization functions *****
*************************************************
'''

def compute_minmax_params(input: Tensor) -> Tuple[Tensor, Tensor]:
    rmin, rmax = torch.min(input, dim=1)[0], torch.max(input, dim=1)[0]
    return rmin, rmax

def integer_quantize(data: Tensor, bits: int, rmin: Tensor, rmax: Tensor, stochastic: bool = True) -> Tuple[Tensor, Tensor]:
    '''
    `input`
        data: shape: [N, F], where N is the batch_size, F is the feature dimension.

        bits: type: int, quantization bit width.

        rmin: shape: [N], min value per node, serve as zero point.

        rmax: shape: [N], max value per node.
    `return`
        q_data: [N/(8/bits)*F]

        scale: [N]
    '''
    assert type(bits) == int
    quant_func = integer_quantizer.pack_single_precision
    scale = (2 ** bits - 1) / (rmax - rmin)  # shape: [N]
    q_data = quant_func(data, rmin, rmax, scale.to(data.dtype), bits, stochastic)
    return q_data, scale

def integer_dequantize(q_data: Tensor, shape: torch.Size, bits: int, scale: Tensor, rmin: Tensor) -> Tensor:
    r'''
    input
        data: shape: [N/(8/bits)*F], where N is the batch_size, bits is the quantization bits,  F is the feature dimension. (already on device)

        shape: the tempinal shape of q_data

        bits: type: int, quantization bit width.

        scale: shape: [N], quantization scale per node. (already on device)

        rmin: shape: [N], min value per node, serve as zero point.

    return
        data: shape: [N, F], where N is the batch_size, F is the feature dimension.
    '''
    N = shape[0]
    num_features = shape[1]
    # unpack bit stream
    assert type(bits) == int
    dequant_func = integer_quantizer.unpack_single_precision
    data = dequant_func(q_data, bits, scale, rmin, N, num_features)
    return data

def message_quantization(input: Tensor, bits: int, stochastic: bool) -> Tuple[Tensor, Tensor, Tensor, torch.Size]:
    rmin, rmax = compute_minmax_params(input)
    q_input, q_scale = integer_quantize(input, bits, rmin, rmax, stochastic=stochastic)
    # transfer with bfloat16
    if input.dtype == torch.float32:
        return q_input, q_scale.to(torch.bfloat16), rmin.to(torch.bfloat16), input.shape
    else:
        return q_input, q_scale, rmin, input.shape

def message_dequantization(q_input: Tensor, q_scale: Tensor, rmin: Tensor, input_tempin_shape: torch.Size, bits):
    if q_scale.dtype == torch.bfloat16:
        q_scale = q_scale.to(torch.float32)
        rmin = rmin.to(torch.float32)
    input = integer_dequantize(q_input, input_tempin_shape, bits, q_scale, rmin)
    return input.contiguous()

'''
*************************************************
***** message exchange functions *****
*************************************************
'''

def trace_input(func):
    @wraps(func)
    def wrapper(send_messages, name, is_train):
        if assigner.ctx.is_tracing:
            rmin, rmax = torch.min(send_messages, dim=1)[0], torch.max(send_messages, dim=1)[0]
            dim = send_messages.shape[1]
            assigner.ctx.traced_layer_data[name] += (dim / 6) * (rmax - rmin) ** 2
        return func(send_messages, name, is_train)
    return wrapper

@trace_input
def msg_all2all_GLOO(local_messages: Tensor, name: str, is_train: bool = True) -> Tensor:
    '''
    perform messages exchange between all workers.
    '''
    assert comm.get_backend() == 'gloo', 'currently only gloo backend is supported'
    # get necessary params
    msg_dim = local_messages.shape[-1]
    msg_dtype = local_messages.dtype
    bit_type = engine.ctx.bit_type
    num_remote = engine.ctx.num_remove
    send_idx = engine.ctx.send_idx
    recv_idx = engine.ctx.recv_idx
    if bit_type == BitType.FULL or not is_train:
        return fp_msg_transfer_process(local_messages, send_idx, recv_idx, msg_dim, msg_dtype, num_remote, name, is_train)
    else:
        return qt_msg_transfer_process(local_messages, send_idx, recv_idx, msg_dim, msg_dtype, num_remote, name)
    

# TODO: add decorator to handle central/marginal graph parallelism
def fp_msg_transfer_process(local_messages: Tensor, send_idx: Dict[int, Tuple[int, int]], recv_idx: Basic_Buffer_Type, msg_dim: int, msg_dtype: torch.dtype, num_remote: int, name: str, is_train: bool) -> Tensor:
    # get communication buffer
    if not is_train:
        idx = int(name[-1])
        recv_buffer_cpu, recv_buffer_gpu, send_buffer_cpu = comm.ctx.comm_buffer.get_test_buffer(idx)
    else:
        recv_buffer_cpu, recv_buffer_gpu, send_buffer_cpu = comm.ctx.comm_buffer.get_train_buffer(name)
    # communication
    with engine.ctx.timer.record(f'{name}_communication'):
        comm.ctx.fp_msg_exchange(recv_buffer_cpu, recv_buffer_gpu, send_buffer_cpu, send_idx, local_messages)
    # reorganize received messages
    remote_messages = torch.zeros(num_remote, msg_dim, dtype=msg_dtype, device=comm.ctx.device)
    for pid, idx in recv_idx.items():
        remote_messages[idx] = recv_buffer_gpu[pid]
    return remote_messages    

def qt_msg_transfer_process(local_messages: Tensor, send_idx: Dict[int, Tuple[int, int]], recv_idx: Basic_Buffer_Type, msg_dim: int, msg_dtype: torch.dtype, num_remote: int, name: str) -> Tensor:
    # get communication buffer and auxillary buffer
    recv_buffer_cpu, recv_buffer_gpu, send_buffer_cpu = comm.ctx.comm_buffer.get_train_buffer(name)
    recv_orig_idx_buffer, recv_orig_size_buffer, send_orig_idx_buffer = comm.ctx.comm_buffer.get_auxillary_buffer(name)
    # quantization
    with engine.ctx.timer.record(f'{name}_quantization'):
        mixed_msg_quantization(local_messages, send_idx, send_buffer_cpu, send_orig_idx_buffer)
    # communication
    with engine.ctx.timer.record(f'{name}_communication'):
        comm.ctx.qt_msg_exchange(recv_buffer_cpu, recv_buffer_gpu, send_buffer_cpu)
    # dequantization
    with engine.ctx.timer.record(f'{name}_de-quantization'):
        remote_messages = mixed_msg_dequantization(recv_idx, recv_buffer_gpu, recv_orig_idx_buffer, recv_orig_size_buffer, msg_dim, msg_dtype, num_remote)
    return remote_messages
    

def mixed_msg_quantization(local_messages: Tensor, send_idx: Dict[int, Tuple[int, int]], send_buffer_cpu: Basic_Buffer_Type, send_orig_idx_buffer: Dict[int, Dict[int, Tensor]]):
    '''
    quantize messages to different bit-widths, and convert them into uniform byte stream for communication.
    '''
    for pid, ids in send_idx.items():
        data = local_messages[ids[0]:ids[1]]
        Q_data = []
        Q_scale = []
        R_min = []
        for bit, local_ids in send_orig_idx_buffer[pid].items():
            q_data, q_scale, r_min, _ = message_quantization(data[local_ids], bit, stochastic=True)
            Q_data.append(q_data)
            Q_scale.append(q_scale)
            R_min.append(r_min)
        Q_data = torch.concat(Q_data)  # [N1+N2+...]
        Q_scale = torch.concat(Q_scale)
        R_min = torch.concat(R_min)
        Q_params = torch.stack([Q_scale, R_min], dim=0)
        send_buffer_cpu[pid][0].copy_(Q_data, non_blocking=True)
        send_buffer_cpu[pid][1].copy_(Q_params, non_blocking=True)

def mixed_msg_dequantization(recv_idx: Basic_Buffer_Type, recv_buffer_gpu: Basic_Buffer_Type, recv_orig_idx_buffer: Dict[int, Dict[int, Tensor]],  recv_orig_size_buffer: Dict[int, Dict[int, Tuple[int, int]]], msg_dim: int, dtype: torch.dtype, num_remote: int) -> Tensor:
    '''
    dequantize received messages to original bit-widths (FP32).
    '''
    remote_tensors = torch.zeros(num_remote, msg_dim, dtype=dtype, device=comm.ctx.device)  # on device
    for pid, ids in recv_idx.items():
        bf_qdata, bf_qparams = recv_buffer_gpu[pid]
        bf_qscale, bf_rmin = bf_qparams[0], bf_qparams[1]
        q_offset = 0
        fp_offset = 0
        sub_remote_tensors = remote_tensors[ids]
        for bit, two_size in recv_orig_size_buffer[pid].items():
            # get proper data
            orig_ids = recv_orig_idx_buffer[pid][bit]
            bit_qdata = bf_qdata[q_offset: q_offset + two_size[0]]
            bit_qscale = bf_qscale[fp_offset: fp_offset + two_size[1]]
            bit_rmin = bf_rmin[fp_offset: fp_offset + two_size[1]]
            shape = torch.Size((len(orig_ids), msg_dim))
            deq_data = message_dequantization(bit_qdata, bit_qscale, bit_rmin, shape, bit)
            sub_remote_tensors[orig_ids] = deq_data  # use local origin indices
            # update indices
            q_offset += two_size[0]
            fp_offset += two_size[1]
        remote_tensors[ids] = sub_remote_tensors
    return remote_tensors
    


def msg_all2all_NCCL(local_messages: Tensor, name: str, is_train: bool = True) -> Tensor:
    raise NotImplementedError
    







