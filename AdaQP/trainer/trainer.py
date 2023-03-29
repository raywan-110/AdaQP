import os
import csv
import yaml
import torch
from argparse import Namespace
from typing import Dict, Tuple

from .runtime_util import *
from ..helper import DistGNNType, BitType
from ..model import DistGCN, DistSAGE
from ..communicator import Communicator as comm
from ..manager import GraphEngine as engine

# supported runnning modes 
# AdaQP: AdaQP with both quantization and parallelization
# AdaQP-q: AdaQP with only quantization
# AdaQP-p: AdaQP with only parallelization
RUNING_MODE = ['Vanilla', 'AdaQP', 'AdaQP-q', 'AdaQP-p']

QUNAT_PARA_MAP: Dict[str, Tuple[str, bool]] = {'Vanilla': ('full', False), 'AdaQP': ('quant', True), 'AdaQP-q': ('quant', False), 'AdaQP-p': ('full', True)}
MODEL_MAP: Dict[str, DistGNNType] = {'gcn': DistGNNType.DistGCN, 'sage': DistGNNType.DistSAGE}

class Trainer(object):
    '''
    the trainer which is responsible for training the model.
    '''
    
    def __init__(self, runtime_args: Namespace):
        # set runtime config
        runtime_args = vars(runtime_args)
        dataset = runtime_args['dataset']
        # load offline config
        current_path = os.path.dirname(__file__)
        offline_config_path = os.path.join(os.path.dirname(current_path), 'config', f'{dataset}.yaml')
        offline_config = yaml.load(open(offline_config_path, 'r'), Loader=yaml.FullLoader)
        # update mode and use_parallel flag according to runtime settings
        offline_config['runtime'].update(runtime_args)
        # set config
        self.config = offline_config
        # set experiment path
        runtime_config = self.config['runtime']
        exp_path = runtime_config['exp_path']
        num_parts = runtime_config['num_parts']
        model_name = runtime_config['model_name']
        log_level = runtime_config['logger_level']
        exp_path = f'{exp_path}/{dataset}/{num_parts}part/{model_name}'
        # set up logger
        self.logger = setup_logger(f'trainer.log', log_level, with_file=True)
        # set up communicator
        self._set_communicator()
        # set up graph engine
        self._set_engine()
        # set exp_path 
        if not os.path.exists(exp_path) and comm.get_rank() == 0:
            os.makedirs(exp_path)
        self.exp_path = exp_path
        # set up comm buffer
        self._set_buffer()
        # set up assigner
        self._set_assigner()
        # first bit-width assignment and update train buffer
        if engine.ctx.bit_type == BitType.QUANT:
            # if use adaptive assignment, use default uniform bit-width assignment when no traced data is available
            if assigner.ctx.scheme == 'adaptive':
                bits_assignments_rst = assigner.ctx.get_assignment(engine.ctx.send_idx, runtime_scheme='uniform')
            else:
                # random sampling scheme / uniform scheme
                bits_assignments_rst = assigner.ctx.get_assignment(engine.ctx.send_idx)
            comm.ctx.update_buffer(bits_assignments_rst)
        # set up model
        self._set_model()
    
    '''
    *************************************************
    ***************** setup methods *****************
    *************************************************
    '''
    
    def _set_communicator(self):
        # fetch corresponding config
        runtime_config = self.config['runtime']
        # setup
        self.communicator = comm(runtime_config['backend'], runtime_config['init_method'])
        self.logger.info(repr(self.communicator))
        
    
    def _set_engine(self):
        # fetch corresponding config
        data_config = self.config['data']
        runtime_config = self.config['runtime']
        model_config = self.config['model']
        if runtime_config['mode'] not in RUNING_MODE:
            raise ValueError(f'Invalid running mode: {runtime_config["mode"]}')
        msg_precision_type, use_parallel = QUNAT_PARA_MAP[runtime_config['mode']]
        if runtime_config['model_name'] not in MODEL_MAP:
            raise ValueError(f'Invalid model type: {model_config["model"]}')
        model_type = MODEL_MAP[runtime_config['model_name']]
        # setup
        self.engine = engine(runtime_config['num_epoches'], data_config['partition_path'], runtime_config['dataset'], msg_precision_type, model_type, use_parallel)
        engine.ctx.agg_type = model_config['aggregator_type']
        if engine.ctx.use_parallel:
            # init the copy buffer
            engine.ctx.graph.init_copy_buffers(data_config['num_feats'], model_config['hidden_dim'], model_config['num_layers'], engine.ctx.device)
            engine.ctx.bwd_graph.init_copy_buffers(data_config['num_feats'], model_config['hidden_dim'], model_config['num_layers'], engine.ctx.device)
        self.logger.info(repr(self.engine))
        
    
    def _set_buffer(self):
        # fetch corresponding config
        data_config = self.config['data']
        model_config = self.config['model']
        buffer_shape = torch.zeros(model_config['num_layers'], dtype=torch.int32)
        buffer_shape[0] = data_config['num_feats']
        buffer_shape[1:] = model_config['hidden_dim']
        # setup
        comm.ctx.init_buffer(buffer_shape.tolist(), engine.ctx.send_idx, engine.ctx.recv_idx, engine.ctx.bit_type)
    
    def _set_assigner(self):
        data_config = self.config['data']
        model_config = self.config['model']
        runtime_config = self.config['runtime']
        assignment_config = self.config['assignment']
        self.assigner = assigner(data_config['num_feats'], model_config['hidden_dim'], model_config['num_layers'], assignment_config['profile_data_length'], runtime_config['assign_scheme'], assignment_config['assign_bits'], engine.ctx.scores, assignment_config['group_size'], assignment_config['coe_lambda'], assignment_config['assign_cycle'])
        self.logger.info(self.assigner)
        pass
    
    def _set_model(self):
        # fetch corresponding config
        data_config = self.config['data']
        model_config = self.config['model']
        runtime_config = self.config['runtime']
        if runtime_config['model_name'] not in MODEL_MAP:
            raise ValueError(f'Invalid model type: {model_config["model"]}')
        model_type = MODEL_MAP[runtime_config['model_name']]
        if model_type == DistGNNType.DistGCN:
            self.model = DistGCN(data_config['num_feats'], model_config['hidden_dim'], data_config['num_classes'], model_config['num_layers'], model_config['dropout_rate'], model_config['use_norm']).to(comm.ctx.device)
        elif model_type == DistGNNType.DistSAGE:
            self.model = DistSAGE(data_config['num_feats'], model_config['hidden_dim'], data_config['num_classes'], model_config['num_layers'], model_config['dropout_rate'], model_config['use_norm'], model_config['aggregator_type']).to(comm.ctx.device)
        else:
            raise ValueError(f'Invalid model type: {model_type}')

    '''
    *************************************************
    **************** runtime methods ****************
    *************************************************
    '''
    
    def train(self):
        # fetch needed config
        runtime_config = self.config['runtime']
        is_multilabel = self.config['data']['is_multilabel']
        # sync seed and model
        sync_seed()
        self.model.reset_parameters()
        sync_model(self.model)
        # set optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=runtime_config['learning_rate'], weight_decay=runtime_config['weight_decay'])
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum') if is_multilabel else torch.nn.CrossEntropyLoss(reduction='sum')
        # record time
        assign_time = []        
        train_time = []
        # get necessary params for training
        epoches = runtime_config['num_epoches']
        input_data = self.engine.ctx.feats
        labels = self.engine.ctx.labels
        train_mask = self.engine.ctx.train_mask
        val_mask = self.engine.ctx.val_mask
        test_mask = self.engine.ctx.test_mask
        # get total number of nodes
        total_number_nodes = torch.LongTensor([train_mask.numel()])
        comm.all_reduce_sum(total_number_nodes)
        total_number_nodes = total_number_nodes.item()
        # enter training loop
        for epoch in range(1, epoches + 1):
            # train for one epoch
            overhead, loss, traced_time, reduce_time = train_for_one_epoch(epoch, engine.ctx.graph, self.model, input_data, labels, optimizer, criterion, total_number_nodes, train_mask)
            # append time records
            assign_time.append(overhead)
            train_time.append(traced_time)
            # validation and test
            epoch_metrics = val_test(engine.ctx.graph, self.model, input_data, labels, train_mask, val_mask, test_mask, is_multilabel)
            metrics_info = aggregate_accuracy(loss, epoch_metrics, epoch) if not is_multilabel else aggregate_F1(loss, epoch_metrics, epoch)
            # print the training information
            if epoch % runtime_config['log_steps'] == 0:
                if comm.get_rank() == 0:
                    if not engine.ctx.use_parallel:
                        time_info = f'Worker {comm.get_rank()} | Total Time {traced_time[0]:.4f}s | Comm Time {traced_time[1]:.4f}s | Quant Time {traced_time[2]:.4f}s | Agg Time {traced_time[-1]:.4f}s | Reduce Time {reduce_time:.4f}s'
                    else:
                        time_info = f'Worker {comm.get_rank()} | Total Time {traced_time[0]:.4f}s | Comm Time {traced_time[1]:.4f}s | Quant Time {traced_time[2]:.4f}s | Central Agg Time {traced_time[3]:.4f}s | Marginal Agg Time {traced_time[4]:.4f}s | Reduce Time {reduce_time:.4f}s'
                    self.logger.info(metrics_info+ '\n' +time_info)
                    comm.barrier()
                else:
                    comm.barrier()
        
        total_epoches_time = torch.tensor(train_time).sum(dim=0)[0]
        assign_time = torch.tensor(assign_time).sum()
        train_time = torch.tensor(train_time).mean(dim=0)
        total_time_records = torch.concat([assign_time.view(-1), total_epoches_time.view(-1), train_time]) # assign_time, total_epoches_time, train_time (breakdown)
        # delete all buffers
        comm.ctx.delete_buffer()
        return total_time_records
    
    def save(self, time_records: Tensor):
        if comm.get_rank() == 0:
            # gather time records
            obj_list = [None] * comm.get_world_size()
            comm.gather_any(time_records, obj_list, dst=0)
            # set up path
            save_path = self.exp_path
            metrics_path = f'{save_path}/metrics'
            time_path = f'{save_path}/time'
            val_curve_path = f'{save_path}/val_curve'
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            if not os.path.exists(val_curve_path):
                os.makedirs(val_curve_path)
            name = self.config['runtime']['mode']
            if engine.ctx.bit_type == BitType.QUANT:
                name = f'{name}_{self.config["runtime"]["assign_scheme"]}'
            # save metrics and val_curve
            engine.ctx.recorder.display_final_statistics(f'{metrics_path}/{name}.txt', f'{val_curve_path}/{name}.pt', self.config['runtime']['model_name'])
            # save time
            set_title = True if not os.path.exists(f'{time_path}/{name}.csv') else False
            with open(f'{time_path}/{name}.csv', 'a') as csvfile:
                writer = csv.writer(csvfile)
                if set_title:
                    writer.writerow(['Worker', 'Overhead', 'Total', 'Per_epoch', 'Comm', 'Quant', 'Central', 'Marginal', 'Full'])
                for worker in range(comm.get_world_size()):
                    write_data = [f'Worker {worker}']
                    write_data.extend(obj_list[worker].numpy())
                    assert len(write_data) == 9, f'Invalid write data length: {len(write_data)}'
                    writer.writerow(write_data)
            comm.barrier()
        else:
            comm.gather_any(time_records, None, dst=0)
            comm.barrier()

    
    
    
    
        
        