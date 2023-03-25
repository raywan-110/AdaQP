import torch
import logging
from torch import Tensor
from typing import Dict, List, Tuple, Union, NewType
import torch.distributed as dist

from ..helper import BitType

logger = logging.getLogger('trainer')
# typing definition
# buffer structure: (pid->messages/(messages, params))
Basic_Buffer_Type = NewType('Basic_Buffer_Type', Dict[int, Union[Tensor, Tuple[Tensor, Tensor]]]) 
# buffer structure: (layer->pid->messages)
Test_Buffer_Type = NewType('Test_Buffer_Type', List[Basic_Buffer_Type])
# buffer structure: (layer->pid->(messages, params))
Train_Buffer_Type = NewType('Train_Buffer_Type', Dict[str, Basic_Buffer_Type])
# buffer structure (layer->pid->bits->original_idx/original_size)
Auxillary_Buffer_Type = NewType('Auxillary_Buffer_Type', Dict[str, Dict[int, Dict[int, Union[Tensor, Tuple[int, int]]]]])

BITS_SET = (2, 4, 8)

class CommBuffer:
    '''
    manage the communication buffer for remote messages exchange.
    '''
    def __init__(self, buffer_shape: List[int], send_idx: Dict[int, Tuple[int, int]], recv_idx: Basic_Buffer_Type, bit_type: BitType, device: torch.device):
        self.buffer_shape = buffer_shape
        self.device = device
        assert bit_type in [BitType.FULL, BitType.QUANT], f'bit_type should be either FULL or QUANT, but got {bit_type}'
        self.bit_type = bit_type
        # buffers for remote messages exchange during validation/testing (use full-presicion messages for testing)
        self.test_recv_buffers_cpu: Test_Buffer_Type = []
        self.test_recv_buffers_gpu: Test_Buffer_Type = []
        self.test_send_buffers_cpu: Test_Buffer_Type = []
        # buffers for remote messages exchange during training
        self.train_recv_buffers_cpu: Train_Buffer_Type = {}
        self.train_recv_buffers_gpu: Train_Buffer_Type = {}
        self.train_send_buffers_cpu: Train_Buffer_Type = {}
        # buffers for recover mixed-precision quantized messages 
        self.send_original_idx_buffers: Auxillary_Buffer_Type = {}
        self.recv_original_idx_buffers: Auxillary_Buffer_Type = {}
        self.recv_original_size_buffers: Auxillary_Buffer_Type = {}
        # generate the buffers for testing
        self._generate_test_buffer(send_idx, recv_idx)
    
    '''
    *************************************************
    ***************** getter methods ****************
    *************************************************
    '''

    def get_test_buffer(self, idx: int) -> Tuple[Basic_Buffer_Type, Basic_Buffer_Type, Basic_Buffer_Type]:
        '''
        get the test buffer ([recv_cpu, recv_gpu, send_cpu]) for validation/testing.
        '''
        return self.test_recv_buffers_cpu[idx], self.test_recv_buffers_gpu[idx], self.test_send_buffers_cpu[idx]
    
    def get_train_buffer(self, layer: str) -> Tuple[Basic_Buffer_Type, Basic_Buffer_Type, Basic_Buffer_Type]:
        '''
        get the train buffer ([recv_cpu, recv_gpu, send_cpu]) for training.
        '''
        if self.bit_type == BitType.FULL:
            idx = int(layer[-1])
            return self.get_test_buffer(idx)
        elif self.bit_type == BitType.QUANT:
            return self.train_recv_buffers_cpu[layer], self.train_recv_buffers_gpu[layer], self.train_send_buffers_cpu[layer]
    
    def get_auxillary_buffer(self, layer: str) -> Tuple[Dict[int, Dict[int, Tensor]], Dict[int, Dict[int, Tuple[int, int]]], Dict[int, Dict[int, Tuple[int, int]]]]:
        '''
        get the auxillary buffer ([recv_orig_idx, recv_size__idx, send_orig_idx]) for training.
        '''
        return self.recv_original_idx_buffers[layer], self.recv_original_size_buffers[layer], self.send_original_idx_buffers[layer]
        
    '''
    *************************************************
    ***************** delete methods ****************
    *************************************************
    '''

    def _delete_buffer_tensors(self, buffer_tensors: Union[Test_Buffer_Type, Train_Buffer_Type, Auxillary_Buffer_Type]):
        '''
        delete the dict of tensors.
        '''
        if isinstance(buffer_tensors, Test_Buffer_Type):
            for buffer_per_layer in range(len(buffer_tensors)):
                for pid in buffer_tensors[buffer_per_layer].keys():
                    buffer_tensors[buffer_per_layer][pid] = None
        elif isinstance(buffer_tensors, Train_Buffer_Type):
            for buffer_per_layer in buffer_tensors.keys():
                for buffer_per_pid in buffer_tensors[buffer_per_layer].keys():
                    buffer_tensors[buffer_per_layer][buffer_per_pid] = None
        elif isinstance(buffer_tensors, Auxillary_Buffer_Type):
            for buffer_per_layer in buffer_tensors.keys():
                for buffer_per_pid in buffer_tensors[buffer_per_layer].keys():
                    for buffer_per_bits in buffer_tensors[buffer_per_layer][buffer_per_pid].keys():
                        buffer_tensors[buffer_per_layer][buffer_per_pid][buffer_per_bits] = None
        else:
            raise TypeError('buffer_tensors should be one of Test_Buffer_Type, Train_Buffer_Type, Auxillary_Buffer_Type')
    
    def _delete_test_buffer(self):
        '''
        delete the test communication buffer.
        '''
        self._delete_buffer_tensors(self.test_recv_buffers_cpu)
        self._delete_buffer_tensors(self.test_recv_buffers_gpu)
        self._delete_buffer_tensors(self.test_send_buffers_cpu)
        self.test_recv_buffers_cpu = []
        self.test_recv_buffers_gpu = []
        self.test_send_buffers_cpu = []
    
    def _delete_train_buffer(self):
        '''
        delete the train communication buffer.
        '''
        # reset idx and size buffer
        self._delete_buffer_tensors(self.send_original_idx_buffers)
        self._delete_buffer_tensors(self.recv_original_idx_buffers)
        self._delete_buffer_tensors(self.recv_original_size_buffers)
        self.send_original_idx_buffers = {}
        self.recv_original_idx_buffers = {}
        self.recv_original_size_buffers = {}
        # reset buffer
        self._delete_buffer_tensors(self.train_recv_buffers_cpu)
        self._delete_buffer_tensors(self.train_recv_buffers_gpu)
        self._delete_buffer_tensors(self.train_send_buffers_cpu)
        self.train_recv_buffers_cpu = {}
        self.train_recv_buffers_gpu = {}
        self.train_send_buffers_cpu = {}
    
    def _delete(self):
        '''
        delete the communication buffer.
        '''
        self._delete_test_buffer()
        self._delete_train_buffer()
        torch.cuda.empty_cache()  # empty the cuda cache
        logger.info(f'<worker{dist.get_rank()} buffer delete done.>')

    '''
    *************************************************
    ***************** setter methods ****************
    *************************************************
    '''
    
    def _generate_test_buffer(self, send_idx: Dict[int, Tuple[int, int]], recv_idx: Basic_Buffer_Type):
        '''
        generate the test communication buffer for validation/testing.
        '''
        # generate the sending buffer
        for dim_size in self.buffer_shape:
            send_temp_buffer: Dict[int, Tensor] = {}
            for pid, idx in send_idx.items():
                num_nodes = idx[1] - idx[0]
                send_temp_buffer[pid] = torch.zeros((num_nodes, dim_size), dtype=torch.float32).pin_memory(self.device)
            self.test_send_buffers_cpu.append(send_temp_buffer)
        # generate the receiving buffer
        for dim_size in self.buffer_shape:
            recv_temp_buffer_cpu: Dict[int, Tensor] = {}
            recv_temp_buffer_gpu: Dict[int, Tensor] = {}
            for pid, idx in recv_idx.items():
                num_nodes = len(idx)
                recv_temp_buffer_cpu[pid] = torch.zeros((num_nodes, dim_size), dtype=torch.float32).pin_memory(self.device)
                recv_temp_buffer_gpu[pid] = torch.zeros((num_nodes, dim_size), dtype=torch.float32, device=self.device)
            self.test_recv_buffers_cpu.append(recv_temp_buffer_cpu)
            self.test_recv_buffers_gpu.append(recv_temp_buffer_gpu)
    
    def _generate_train_buffer(self, assigned_bits_results: Auxillary_Buffer_Type, bits: Tuple[int, ...] = BITS_SET):
        '''
        generate the train communication buffer for training.
        '''
        # internal func to get quantized tensor size
        def get_qsize(number_nodes: int, bits: int, feat_dim: int) -> int:
            work_per_thread = int(8 / bits)
            N = number_nodes + (work_per_thread - number_nodes % work_per_thread) % work_per_thread
            total_bits = bits * (N * feat_dim)
            q_size = int((total_bits + 8) / 8)
            return q_size
        # generate the sending original idx buffer and sending size idx buffer (temp)
        temp_send_origin_size_buffer: Dict[str, Dict[int, Dict[int, Tensor]]] = {}
        for layer, config_per_layer in assigned_bits_results.items():
            self.send_original_idx_buffers[layer] = {}
            temp_send_origin_size_buffer[layer] = {}
            for pid, config_per_pid in config_per_layer.items():
                self.send_original_idx_buffers[layer][pid] = {}
                temp_send_origin_size_buffer[layer][pid] = {}
                for b in bits:
                    b_ids = torch.nonzero(config_per_pid == b).view(-1)  # local ids for different bits
                    # skip the empty bits
                    if len(b_ids) == 0:
                        continue
                    else:
                        layer_idx = int(layer[-1]) # the layer index (e.g., 0,1,2)
                        b_qsize = get_qsize(len(b_ids), b, self.buffer_shape[layer_idx])
                        self.send_original_idx_buffers[layer][pid][b] = b_ids
                        temp_send_origin_size_buffer[layer][pid][b] = (b_qsize, len(b_ids)) # (qt_size, fp_data_size)
        # generate the sending buffer
        for layer in self.send_original_idx_buffers.items():
            self.train_send_buffers_cpu[layer] = {}
            for pid, size_buffer in temp_send_origin_size_buffer[layer].items():
                layer_qt_size = 0
                layer_fp_size = 0
                # aggregate the total size for each layer according to bits order (e.g., [2,4,8])
                for qt_fp_size in size_buffer.values():
                    layer_qt_size += qt_fp_size[0]
                    layer_fp_size += qt_fp_size[1]
                    send_qdata = torch.zeros(size=(layer_qt_size,), dtype=torch.int8).pin_memory(device=self.device)
                    send_qparams = torch.zeros(size=(2, layer_fp_size), dtype=torch.bfloat16).pin_memory(device=self.device)
                    self.train_send_buffers_cpu[layer][pid] = (send_qdata, send_qparams)
        # generate the receiving original idx buffer and receiving size idx buffer (temp)
        rank, world_size = dist.get_rank(), dist.get_world_size()
        sending_origin_idx_size_buffer_list = [None for _ in range(world_size)]
        dist.all_gather_object(sending_origin_idx_size_buffer_list, [self.send_original_idx_buffers, temp_send_origin_size_buffer])
        layer_keys = self.send_original_idx_buffers.keys()
        for layer in layer_keys:
            self.recv_original_idx_buffers[layer] = {}
            self.recv_original_size_buffers[layer] = {}
            for i in range(world_size):
                # if worker i needs to send data to current worker rank
                if i is not rank and rank in sending_origin_idx_size_buffer_list[i][0].keys():
                    self.recv_original_idx_buffers[layer][i] = sending_origin_idx_size_buffer_list[i][0][rank]  # get original idx buffer from worker i
                    self.recv_original_size_buffers[layer][i] = sending_origin_idx_size_buffer_list[i][1][rank]  # get original size buffer from worker i
        # generate the receiving buffer
        for layer in layer_keys:
            self.train_recv_buffers_cpu[layer] = {}
            self.train_recv_buffers_gpu[layer] = {}
            for pid, size_buffer in self.recv_original_size_buffers[layer].items():
                layer_qt_size = 0
                layer_fp_size = 0
                # aggregate the total size for each layer according to bits order (e.g., [2,4,8])
                for qt_fp_size in size_buffer.values():
                    layer_qt_size += qt_fp_size[0]
                    layer_fp_size += qt_fp_size[1]
                    recv_qdata_cpu = torch.zeros(size=(layer_qt_size,), dtype=torch.int8).pin_memory(device=self.device)
                    recv_qparams_cpu = torch.zeros(size=(2, layer_fp_size), dtype=torch.bfloat16).pin_memory(device=self.device)
                    recv_qdata_gpu = torch.zeros(size=(layer_qt_size,), dtype=torch.int8).to(device=self.device)
                    recv_qparams_gpu = torch.zeros(size=(2, layer_fp_size), dtype=torch.bfloat16).to(device=self.device)
                    self.train_recv_buffers_cpu[layer][pid] = (recv_qdata_cpu, recv_qparams_cpu)
                    self.train_recv_buffers_gpu[layer][pid] = (recv_qdata_gpu, recv_qparams_gpu)

    '''
    *************************************************
    *************** external interface **************
    *************************************************
    '''

    def _update(self, *args, **kwargs):
        '''
        update the communication buffer for training.
        '''
        if self.bit_type == BitType.FULL:
            pass
        else:
            self._generate_train_buffer(*args, **kwargs)
            logging.info(f'<worker {dist.get_rank()} buffer update done>')
