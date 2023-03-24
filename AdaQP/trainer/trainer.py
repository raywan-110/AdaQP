import os
import csv
import yaml
import torch
from argparse import Namespace
from typing import Dict, Tuple

from .runtime_util import *
from ..model import DistGCN, DistSAGE, DistGNNType
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
        offline_config_path = os.path.join(os.path.dirname(__file__), 'config', f'{dataset}.yaml')
        offline_config = yaml.load(open(offline_config_path, 'r'), Loader=yaml.FullLoader)
        # update mode and use_parallel flag according to runtime settings
        
        offline_config['runtime'].update(runtime_args)
        # set config
        self.config = offline_config
        # set experiment path
        exp_path = self.config['runtime']['exp_path']
        num_parts = self.config['runtime']['num_parts']
        model_name = self.config['runtime']['model_name']
        exp_path = f'{exp_path}/{dataset}/{num_parts}part/{model_name}'
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        self.exp_path = exp_path
        # set up logger
        self.logger = setup_logger('trainer_logger', f'{exp_path}/trainer.log', with_file=True)
        # set up communicator
        self._set_communicator()
        # set up graph engine
        self._set_engine()
        # set up comm buffer
        self._set_buffer()
        # TODO set up assigner
        self.assigner = None
        # TODO update comm buffer according to the assigner
        comm.ctx.update_buffer()
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
        self.communicator = comm(runtime_config['local_world_size'], runtime_config['backend'], runtime_config['init_method'])
        logging.info(self.communicator)
        
    
    def _set_engine(self):
        # fetch corresponding config
        data_config = self.config['data']
        runtime_config = self.config['runtime']
        model_config = self.config['model']
        if runtime_config['mode'] not in RUNING_MODE:
            raise ValueError(f'Invalid running mode: {runtime_config["mode"]}')
        msg_precision_type, use_parallel = QUNAT_PARA_MAP[runtime_config['mode']]
        # setup
        self.engine = engine(runtime_config['num_epoches'], data_config['partition_path'], data_config['dataset_path'], msg_precision_type, use_parallel)
        engine.ctx.agg_type = model_config['aggregator_type']
        logging.info(self.engine)
    
    def _set_buffer(self):
        # fetch corresponding config
        data_config = self.config['data']
        model_config = self.config['model']
        buffer_shape = [None for _ in range(len(model_config['num_layers']))]
        buffer_shape[0] = data_config['num_feats']
        buffer_shape[1:] = model_config['hidden_dim']
        # setup
        comm.ctx.init_buffer(buffer_shape, engine.ctx.send_idx, engine.ctx.recv_idx, engine.ctx.bit_type)
    
    def _set_assigner(self):
        self.assigner = None
        logging.info(self.assigner)
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
            self.model = DistGCN(data_config['num_feats'], model_config['hidden_dim'], model_config['num_classes'], model_config['num_layers'], model_config['dropout_rate'], model_config['use_norm']).to(comm.ctx.device)
        elif model_type == DistGNNType.DistSAGE:
            self.model = DistSAGE(data_config['num_feats'], model_config['hidden_dim'], model_config['num_classes'], model_config['num_layers'], model_config['dropout_rate'], model_config['use_norm'], model_config['aggregator_type']).to(comm.ctx.device)
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
        sync_model()
        # set optimizer and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=runtime_config['lr'], weight_decay=runtime_config['weight_decay'])
        criterion = torch.nn.BCEWithLogitsLoss() if is_multilabel else torch.nn.CrossEntropyLoss()
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
        for epoch in range(1, epoches + 1):
            # TODO use decorator to handle bit-width reassignment
            # train for one epoch
            loss, traced_time, reduce_time = train_for_one_epoch(engine.ctx.graph, self.model, input_data, labels, optimizer, criterion, total_number_nodes, train_mask)
            train_time.append(traced_time)
            # validation and test
            epoch_metrics = val_test(engine.ctx.graph, self.model, input_data, labels, train_mask, val_mask, test_mask, is_multilabel)
            metrics_info = aggregate_accuracy(loss, epoch_metrics) if not is_multilabel else aggregate_F1(loss, epoch_metrics)
            if epoch % runtime_config['log_steps'] == 0:
                if not engine.ctx.use_parallel:
                    time_info = f'Worker {comm.get_rank()} | Total Time {traced_time[0]:.4f}s | Comm Time {traced_time[1]:.4f}s | Quant Time {traced_time[2]:.4f}s | Agg Time {traced_time[-1]:.4f}s | Reduce Time {reduce_time:.4f}s'
                else:
                    time_info = f'Worker {comm.get_rank()} | Total Time {traced_time[0]:.4f}s | Comm Time {traced_time[1]:.4f}s | Quant Time {traced_time[2]:.4f}s | Central Agg Time {traced_time[3]:.4f}s | Marginal Agg Time {traced_time[4]:.4f}s | Reduce Time {reduce_time:.4f}s'
                logging.info(metrics_info+ '\n' +time_info)
        
        total_epoches_time = torch.tensor(train_time).sum(dim=0)[0]
        assign_time = torch.tensor(assign_time).sum()
        total_time_records = torch.concat([assign_time.view(-1), total_epoches_time.view(-1), train_time]) # assign_time, total_epoches_time, train_time (breakdown)
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
                os.makedirs(save_path)
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            if not os.path.exists(val_curve_path):
                os.makedirs(val_curve_path)
            mode = self.config['runtime']['mode']
            # save metrics and val_curve
            engine.ctx.recorder.display_final_statistics(f'{metrics_path}/{mode}.txt', f'{val_curve_path}/{mode}.pt', self.config['runtime']['model_name'])
            # save time
            set_title = True if not os.path.exists(f'{time_path}/{mode}.csv') else False
            with open(f'{time_path}/{mode}.csv', 'a') as csvfile:
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

    
    
    
    
        
        