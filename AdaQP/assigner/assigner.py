import time 
import logging
import torch
from typing import Dict, List, Tuple, Union
from itertools import chain
from multiprocessing.pool import ThreadPool
from queue import Queue
from torch import Tensor
import numpy as np
import pulp as plp
from .profile import *
from ..communicator import BITS_SET
from ..communicator import Communicator as comm

logger = logging.getLogger('trainer')

ASSIGNMENT_SCHEME = ('uniform', 'random', 'adaptive')

class Assigner(object):
    '''
    bit-width assigner, assigning bit-width for each message group.
    '''
    def __init__(self, feat_dim: int, hidden_dim: int, num_layers: int, num_data: int, scheme: str, uniform_assign_bits: int, scores: Dict[int, Tuple[Tensor, Tensor]], group_size: int, coe_lambda: float, assign_cycle: int = None, warmup: int = 1):
        # basic parameters
        self.bits_set = torch.tensor(BITS_SET, dtype=torch.int32)
        self.bits_cost = torch.tensor([1 / (2 ** bit - 1) ** 2 for bit in BITS_SET], dtype=torch.float32)
        assert scheme in ASSIGNMENT_SCHEME, f'assignment scheme {scheme} is not supported'
        # profile related parameters
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_data = num_data
        self.warmup = warmup
        # schemes related parameters
        self._scheme = scheme
        self._scheme_map = {'uniform': self._get_uniform_assignment, 'random': self._get_random_sampling_assignment, 'adaptive': self._get_adaptive_assignment}
        self.num_layers = num_layers
        self.uniform_assign_bits = uniform_assign_bits
        self.scores = scores
        self.group_size = group_size
        self.coe_lambda = coe_lambda
        self.assign_cycle = assign_cycle # assign bit-width every assign_cycle steps, only valid in adaptive scheme
        # cost model related parameters
        self.dataset = None
        self.cost_model = None
        # bit-width assignment schemes related parameters
        self.sample_rate = torch.tensor([ 1 / len(self.bits_set) for _ in range(len(self.bits_set))]) # uniform sampling
        # data tracing related parameters
        self.is_tracing = False
        self.traced_layer_data: Dict[str, Union[Tensor, Dict[int, Tensor]]] = {}
        # objective solving related parameters
        self.normalization_mode = ('magnitude', 'nadir_utopia')
        self.solving_pool = None
        self.group_idx: Dict[str, Dict[int, Tensor]] = {}
        if scheme == 'adaptive':
            self._init_adaptive()
        # set context
        Assigner.ctx = self
    
    def _init_adaptive(self):
        '''
        profile to get the communication cost model of each worker pair.
        '''
        logger.info(f'<worker {comm.get_rank()} preprocessing for adaptive bit-width assignment...>')
        self.dataset = generate_cost_model_dataset(self.feat_dim, self.hidden_dim, self.num_data, self.warmup)
        self.cost_model = fit_cost_model(self.dataset)
        self.dataset = None
        # set tracing state and init traced data if needed  
        self.is_tracing = True
        self.init_traced_data(self.num_layers)
        self.solving_pool = ThreadPool(processes= 2 * self.num_layers - 1)

    def get_assignment(self, send_idx: Dict[int, Tuple[int, int]], runtime_scheme: str = None) -> Dict[str, Dict[int, Tensor]]:
        if runtime_scheme is None:
            return self._scheme_map[self._scheme](send_idx)
        else:
            assert runtime_scheme in ASSIGNMENT_SCHEME, f'assignment scheme {runtime_scheme} is not supported'
            return self._scheme_map[runtime_scheme](send_idx)
    
    def __repr__(self):
        return f'<Assigner(rank: {comm.get_rank()}, default scheme={self._scheme})>'
    
    @property
    def scheme(self):
        return self._scheme
    
    '''
    *************************************************
    ************* simple scheme methods *************
    *************************************************
    '''
    
    def _get_uniform_assignment(self, send_idx: Dict[int, Tuple[int, int]]) -> Dict[str, Dict[int, Tensor]]:
        bits_assignments_rst: Dict[str, Dict[int, Tensor]] = {}
        # initialize the buffer 
        for i in range(self.num_layers):
            bits_assignments_rst[f'forward{i}'] = {}
        for i in range(1, self.num_layers):
                bits_assignments_rst[f'backward{i}'] = {}
        # assign bit-width
        for layer_key in bits_assignments_rst.keys():
            for pid, idx in send_idx.items():
                bits_assignments_rst[layer_key][pid] = torch.ones(size=(idx[1] - idx[0],), dtype=torch.int32) * self.uniform_assign_bits
        return bits_assignments_rst
    
    def _get_random_sampling_assignment(self, send_idx: Dict[int, Tuple[int, int]]) -> Dict[str, Dict[int, Tensor]]:
        bits_assignments_rst: Dict[str, Dict[int, Tensor]] = {}
        # initialize the buffer 
        for i in range(self.num_layers):
            bits_assignments_rst[f'forward{i}'] = {}
        for i in range(1, self.num_layers):
                bits_assignments_rst[f'backward{i}'] = {}
        # assign bit-width
        for layer_key in bits_assignments_rst.keys():
            for pid, idx in send_idx.items():
                idx = torch.multinomial(self.sample_rate, idx[1] - idx[0], replacement=True)
                bits_assignments_rst[layer_key][pid] = self.bits_set[idx]
        return bits_assignments_rst

    '''
    *************************************************
    ************ adaptive scheme methods ************
    *************************************************
    '''
    
    def _get_adaptive_assignment(self, send_idx: Dict[int, Tuple[int, int]]) -> Dict[str, Dict[int, Tensor]]:
        bits_assignments_rst: Dict[str, Dict[int, Tensor]] = {}
        # slice traced data along worker pair
        self.slice_traced_data(send_idx)
        # get var matrix and comm cost matrix
        var_matrix, comm_cost_matrix = self.config_score_matrix(self.scores, self.group_size, self.feat_dim, self.hidden_dim)
        # get group assignment
        group_assignments = self.aggregate_params_get_solution(var_matrix, comm_cost_matrix, self.coe_lambda)
        # recover assignment from group assignment
        bits_assignments_rst = self.recover_assignment_from_group(group_assignments)
        # clear traced data and group idx
        for layer_key in self.traced_layer_data.keys():
            self.traced_layer_data[layer_key] = 0.0
        self.group_idx.clear()
        return bits_assignments_rst
    
    def init_traced_data(self, num_layers: int):
        for i in range(num_layers):
            self.traced_layer_data[f'forward{i}'] = 0.0
        for i in range(1, num_layers):
            self.traced_layer_data[f'backward{i}'] = 0.0    
    
    def slice_traced_data(self, send_idx: Dict[int, Tuple[int, int]]):
        sliced_traced_data: Dict[str, Dict[int, Tensor]] = {}
        for layer_key, data_per_layer in self.traced_layer_data.items():
            sliced_traced_data[layer_key] = {}
            data_per_layer = data_per_layer.cpu()
            for pid, idx in send_idx.items():
                sliced_traced_data[layer_key][pid] = data_per_layer[idx[0]:idx[1]]
        # clear and reser traced data value
        self.traced_layer_data.clear()
        torch.cuda.empty_cache()
        self.traced_layer_data = sliced_traced_data
    
    def config_score_matrix(self, scores: Dict[int, Tuple[Tensor, Tensor]], group_size: int, feats_dim: int, hidden_dim: int) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        '''
        config variance matrix (after grouping), comm matrix (after grouping), store the group idx for recover the bits assignment for each messages.
        '''
        
        def group_data(scores: Tensor, group_size: int) -> Tuple[Tensor, Tuple[Tensor]]:
            sorted_score, idx = torch.sort(scores, descending=True)
            g_ids = torch.split(idx, group_size)
            group_score = torch.tensor([sorted_score[idx].sum() for idx in g_ids])
            return group_score, g_ids
        
        rank = comm.get_rank()
        # config variance matrix for each p2p channel in each layer
        var_matrix: Dict[str, Dict[str, np.ndarray]] = {}
        idx_set: Dict[str, Dict[str, np.ndarray]] = {}
        for layer_key, data_per_layer in self.traced_layer_data.items():
            var_matrix[layer_key] = {}
            idx_set[layer_key] = {}
            for pid, data_per_rank in data_per_layer.items():
                # get combined variance agg_score
                if 'forward' in layer_key:
                    agg_score = scores[pid][0] # forward scores
                else:
                    agg_score = scores[pid][1] # backward scores
                assert agg_score.shape == data_per_rank.shape
                combined_var = (agg_score ** 2) * data_per_rank
                # group data to reduce the magnitude of variables 
                group_var, group_ids = group_data(combined_var, group_size)
                group_var = torch.matmul(self.bits_cost.view(-1, 1), group_var.view(1, -1))  # [N, k]
                # the key format is '{sender}_{receiver}}'
                var_matrix[layer_key][f'{rank}_{pid}'] = group_var.numpy()
                idx_set[layer_key][pid] = group_ids
        self.group_idx = idx_set
        # config comm matrix for each p2p channel in each layer
        layer_keys = self.traced_layer_data.keys()
        comm_matrix: Dict[str, Dict[str, np.ndarray]] = {}
        for layer_key in layer_keys:
            pid_keys = self.traced_layer_data[layer_key].keys()
            comm_matrix[layer_key] = {}
            for pid in pid_keys:
                # get communication cost
                group_length = len(idx_set[layer_key][pid])
                comm_cost = torch.repeat_interleave(self.bits_set.view(-1, 1).float(), group_length, dim=1)  # [N, K]
                # get communication time
                if '0' in layer_key:
                    comm_data = (comm_cost * feats_dim * group_size) / 8 / (1024 ** 2)  # MB
                else:
                    comm_data = (comm_cost * hidden_dim * group_size) / 8 / (1024 ** 2)
                # store the results
                comm_matrix[layer_key][f'{rank}_{pid}'] = comm_data.numpy()
        return var_matrix, comm_matrix
    
    def aggregate_params_get_solution(self, var_matrix: Dict[str, Dict[str, np.ndarray]], comm_matrix: Dict[str, Dict[str, np.ndarray]], coe_lambda: float) -> Dict[str, Dict[int, Tensor]]:
        '''
        master worker gather parameters from others, and then solving the bi-objective optimization problem to get the assignment for message groups.
        '''
        
        # define concat params function
        def rearrange_concat_params(params_list, world_size: int) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
            '''
            concat params from different workers to layer order.
            '''
            all_layer_var: Dict[str, Dict[str, np.ndarray]] = {}
            all_layer_comm: Dict[str, Dict[str, np.ndarray]] = {}
            layer_keys = params_list[0][0].keys()
            # aggregate var and comm matrix for each layer
            for layer_key in layer_keys:
                per_layer_var, per_layer_comm  = [], []
                # gather params in current layer
                for i in range(world_size):
                    per_layer_var.append(params_list[i][0][layer_key].items())
                    per_layer_comm.append(params_list[i][1][layer_key].items())                
                # concat all the parameters    
                all_layer_var[layer_key] = dict(chain(*per_layer_var))
                all_layer_comm[layer_key] = dict(chain(*per_layer_comm))
            # aggregate cost model for each channel
            all_layer_model = []
            for i in range(world_size):
                all_layer_model.append(params_list[i][-1].items())
            all_layer_model = dict(chain(*all_layer_model))
            return all_layer_var, all_layer_comm, all_layer_model
        
        # define scatter params function
        def rearrange_scatter_assignments(layer_assignments: Dict[str, Dict[str, Tensor]], world_size: int) -> List[Dict[str, Dict[int, Tensor]]]:
            '''
            scatter results from master to workers (the key format is also transformed from '{sender}_{receiver} to int(receiver) for each worker).
            '''
            assignments_list = []
            for sender in range(world_size):
                sender_bits_rst: Dict[str, Dict[int, Tensor]] = {}
                for layer_key, data_per_layer in layer_assignments.items():
                    sender_bits_rst[layer_key] = {}
                    for channel_key, data_per_channel in data_per_layer.items():
                        # match the sender in the channel key
                        channel_sender, channel_recveiver = channel_key.split('_')
                        if sender == int(channel_sender):
                            sender_bits_rst[layer_key][int(channel_recveiver)] = data_per_channel
                assignments_list.append(sender_bits_rst)  # in rank order
            return assignments_list
        
        rank, world_size = comm.get_rank(), comm.get_world_size()
        # gather parameters
        params = [var_matrix, comm_matrix, self.cost_model]
        params_list = [None] * world_size
        comm.gather_any(params, params_list if rank == 0 else None, dst=0)
        # solve the bi-objective optimization problem
        if rank != 0:
            assignments_list = [None] * world_size
            comm.barrier()
        else:
            layer_keys = var_matrix.keys()
            all_layer_var, all_layer_comm, all_layer_model = rearrange_concat_params(params_list, world_size)
            # set solving pools
            layer_assignments: Dict[str, Dict[str, Tensor]] = {}
            layer_solve_handle = Queue()
            for layer_key in layer_keys:
                handle = self.solving_pool.apply_async(self.get_solution, args=(all_layer_var[layer_key], all_layer_comm[layer_key], all_layer_model, coe_lambda, world_size))
                layer_solve_handle.put((layer_key, handle))
            # wait for solution
            while not layer_solve_handle.empty():
                layer_key, handle = layer_solve_handle.get()
                solving_time, assignment = handle.get()
                layer_assignments[layer_key] = assignment
                logger.info(f'layer {layer_key} solving time: {solving_time:.4f}s')
            # scatter assignments
            assignments_list = rearrange_scatter_assignments(layer_assignments, world_size)
            comm.barrier()
        # dispatch assignments to each worker
        rank_assignment = [None]
        comm.scatter_any(rank_assignment, assignments_list, src=0)
        return rank_assignment[0]

    def recover_assignment_from_group(self, group_assignments: Dict[str, Dict[int, Tensor]]) -> Dict[str, Dict[int, Tensor]]:
        bits_assignments_rst: Dict[str, Dict[int, Tensor]] = {}
        for layer_key, data_per_layer in self.traced_layer_data.items():
            bits_assignments_rst[layer_key] = {}
            # recover bits allocation reults from group rst
            for rank_key, data_per_rank in data_per_layer.items():
                alloc_rank_bits = torch.zeros_like(data_per_rank, dtype=torch.int32)
                for idx, bits in zip(self.group_idx[layer_key][rank_key], group_assignments[layer_key][rank_key]):
                    alloc_rank_bits[idx] = bits
                bits_assignments_rst[layer_key][rank_key] = alloc_rank_bits
        return bits_assignments_rst

    '''
    *************************************************
    *********** solving & dispatch methods ************
    *************************************************
    '''

    def get_solution(self, var_matrix: Dict[str, Dict[str, np.ndarray]], comm_matrix: Dict[str, Dict[str, np.ndarray]], cost_model: Dict[str, np.ndarray], coe_lambda: float, world_size: int, normal_mode: str = 'nadir_utopia'):
        '''
        invode solver to get the solution.
        '''
        
        def get_scaling_factor(var_matrix: Dict[str, Dict[str, np.ndarray]], comm_matrix: Dict[str, Dict[str, np.ndarray]], cost_model: Dict[str, np.ndarray], normal_mode: str, world_size: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
            # magnitude normalization
            if normal_mode == 'magnitude':
                # get magnitude for variance objective
                max_var = 0
                for _, data_per_channel in var_matrix.items():
                    max_var += sum(data_per_channel[0])  # all groups all assigned with 2 bits
                max_comm_time = 0
                # find max comm time in each round and sum the max comm time in all rounds to get the magnitude
                # from round 1 -> round world_size - 1
                for round in range(1, world_size):
                    round_max_comm_time = 0
                    for rank in range(world_size):
                        dst = (rank + round) % world_size
                        channel_key = f'{rank}_{dst}'
                        max_current_channel = cost_model[channel_key][0] * sum(comm_matrix[channel_key][-1]) + cost_model[channel_key][1]
                        if round_max_comm_time < max_current_channel:
                            round_max_comm_time = max_current_channel  # all groups are assigned with 8 bits
                    max_comm_time += round_max_comm_time
                return max_var, max_comm_time
            else:
                # get nadir and utopia solutions for variance objective
                var_nadir, var_utopia = 0.0, 0.0
                for _, data_per_channel in var_matrix.items():
                    var_nadir += sum(data_per_channel[0])  # maximum variance
                    var_utopia += sum(data_per_channel[-1])  # minimum variance
                # get nadir and utopia values for time objective
                time_nadir, time_utopia = 0.0, 0.0
                # get nadir and utopia values for each communication round
                # from round 1 -> round world_size - 1
                for round in range(1, world_size):
                    round_time_nadir, round_time_utopia = float('-inf'), float('inf')
                    for rank in range(world_size):
                        dst = (rank + round) % world_size
                        channel_key = f'{rank}_{dst}'
                        # all groups are assigned with 8 bits
                        nadir_current_channel = cost_model[channel_key][0] * sum(comm_matrix[channel_key][-1]) + cost_model[channel_key][1]
                        # all groups are assigned with 2 bits
                        utopia_current_channel = cost_model[channel_key][0] * sum(comm_matrix[channel_key][0]) + cost_model[channel_key][1]
                        if round_time_nadir < nadir_current_channel:
                            round_time_nadir = nadir_current_channel
                        if round_time_utopia > utopia_current_channel:
                            round_time_utopia = utopia_current_channel
                    time_nadir += round_time_nadir
                    time_utopia += round_time_utopia
                return (var_nadir, var_utopia), (time_nadir, time_utopia)
        
        def add_constraint(opt_model: plp.LpProblem, var_comm_vars: Dict[Tuple[int, int], plp.LpVariable], comm_matrix: Dict[str, Dict[str, np.ndarray]], Z: List[plp.LpVariable], cost_model: Dict[str, np.ndarray], size_buffer: Dict[str, List[int]], world_size: int):
            # add constraint for auxillary variables [Z_i] for each comm round
            for round in range(1, world_size):
                for rank in range(world_size):
                    dst = (rank + round) % world_size
                    channel_key = f'{rank}_{dst}'
                    data_per_channel = comm_matrix[channel_key]
                    channel_cost_model = cost_model[channel_key]
                    channel_vars = var_comm_vars[channel_key]
                    channel_size = size_buffer[channel_key]
                    channel_constraint = []
                    channel_constraint.extend([channel_vars[i, j] * data_per_channel[i, j] * channel_cost_model[0] for i in range(channel_size[0]) for j in range(channel_size[1])])  # add the output comm cost of the cost model 
                    channel_constraint.extend([channel_cost_model[1], -1 * Z[round - 1]])  # add latency and auxillary variable Z_i for round i
                    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(channel_constraint), sense=plp.LpConstraintLE, rhs=0))
            # add constraint for binary decision variables (each group can only be assigned with one type of bits)
            for channel_key, var_per_channel in var_comm_vars.items():
                size_channel = size_buffer[channel_key]
                for j in range(size_channel[1]):
                    opt_model.addConstraint(plp.LpConstraint(e=plp.lpSum(var_per_channel[i, j] for i in range(size_channel[0])), sense=plp.LpConstraintEQ, rhs=1))
        
        # get normalization factor
        assert normal_mode in self.normalization_mode, f'normalization mode {normal_mode} is not supported'
        scale_var, scale_time = get_scaling_factor(var_matrix, comm_matrix, cost_model, normal_mode, world_size)
        # init variables
        num_bits_options = len(self.bits_set)
        var_comm_vars: Dict[Tuple[int, int], plp.LpVariable] = {}
        size_buffer: Dict[str, List[int, int]] = {}
        for channel_key, data_per_channel in var_matrix.items():
            # set variables for current channel, x_{i, j} means allocate bits i to group j
            num_worker_option = data_per_channel.shape[-1]
            size_buffer[channel_key] = [num_bits_options, num_worker_option]
            vars = {(i, j): plp.LpVariable(cat=plp.LpBinary, name=f'{channel_key}_x_{i}_{j}') for i in range(num_bits_options) for j in range(num_worker_option)}
            var_comm_vars[channel_key] = vars
        # init auxillary variables Z_i for each round
        Z = [plp.LpVariable(cat=plp.LpContinuous, name=f'Z_{i}') for i in range(1, world_size)]
        # define the problem and add constraints
        opt_model = plp.LpProblem(name='MIP_Model', sense=plp.const.LpMinimize)
        add_constraint(opt_model, var_comm_vars, comm_matrix, Z, cost_model, size_buffer, world_size)
        # set objectives
        total_var = []
        for channel_key, data_per_channel in var_matrix.items():
            channel_var = var_comm_vars[channel_key]
            channel_size = size_buffer[channel_key]
            total_var.extend(channel_var[i, j] * data_per_channel[i, j] for i in range(channel_size[0]) for j in range(channel_size[1]))
        if normal_mode == 'magnitude':
            objective = coe_lambda * plp.lpSum(total_var) / scale_var + (1 - coe_lambda) * plp.lpSum(Z) / scale_time  # variance & time objectives Scalarization
        else:
            objective = coe_lambda * (plp.lpSum(total_var) - scale_var[-1]) / (scale_var[0] - scale_var[-1]) + (1 - coe_lambda) * (plp.lpSum(Z) - scale_time[-1]) / (scale_time[0] - scale_time[-1])
        opt_model.setObjective(objective)
        # solve the problem
        avaliable_solvers = plp.list_solvers(onlyAvailable=True)
        start = time.time()
        if 'GUROBI' in avaliable_solvers:
            opt_model.solve(plp.GUROBI(msg=False))
        else:
            opt_model.solve(plp.PULP_CBC_CMD(msg=False))
        solving_time = time.time() - start
        # get the optimal solution
        bits_assignment = {} 
        for channel_key, data_per_channel in var_comm_vars.items():
            channel_size = size_buffer[channel_key]
            channel_rst = torch.zeros(channel_size[-1], dtype=torch.int32)
            x_vars_tensor = torch.tensor([x.value() for x in data_per_channel.values()]).view(channel_size[0], channel_size[1])  # [N, K]
            for i in range(x_vars_tensor.shape[0]):
                idx = torch.nonzero(x_vars_tensor[i])
                channel_rst[idx] = self.bits_set[i]  # the group rst
            bits_assignment[channel_key] = channel_rst
        return solving_time, bits_assignment
        

        
    
        
        