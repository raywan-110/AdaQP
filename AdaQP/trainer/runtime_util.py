import logging
import time
import torch
from typing import Any, List, Tuple, Union
from dgl import DGLHeteroGraph
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
import numpy as np

from ..helper import BitType
from ..communicator import Communicator as comm
from ..manager import GraphEngine as engine
from ..assigner import Assigner as assigner

'''
*************************************************
**************** setup functions ****************
*************************************************
'''
    
def setup_logger(log_file, level=logging.INFO, with_file=True):
    """Function setup as many loggers as you want"""
    config_logger = logging.getLogger('trainer')
    config_logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # file handler
    if with_file:
        file_handler = logging.FileHandler(log_file)        
        file_handler.setFormatter(formatter)
        config_logger.addHandler(file_handler)
    return config_logger

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def sync_seed():
    '''
    synchroneize the seed of all workers
    '''
    if comm.get_rank() == 0:
        seed = int(time.time() % (2 ** 32 - 1))
        obj_list = [seed]
        comm.broadcast_any(obj_list, src = 0)
        fix_seed(seed)
    else:
        obj_list = [None]
        comm.broadcast_any(obj_list, src = 0)
        seed = obj_list[0]
        fix_seed(seed)

def sync_model(model: nn.Module):
    '''
    synchronize the model parameters of all workers
    '''
    state_dict = model.state_dict()
    for _, s_value in state_dict.items():
        if comm.get_rank() != 0:
            s_value.zero_()
        comm.all_reduce_sum(s_value.data)

'''
*************************************************
*************** runtime functions ***************
*************************************************
'''

def average_gradients(model: nn.Module):
    '''
    average the gradients of all workers
    '''
    for _, param in model.named_parameters():
        if param.requires_grad:
            comm.all_reduce_sum(param.grad.data)


def train_for_one_epoch(epoch: int, graph: DGLHeteroGraph, model: nn.Module, input_data: Tensor, labels: Tensor, optimizer: Optimizer, criterion: Union[nn.Module, Any], total_num_training_samples: int, train_mask: Tensor) -> Tuple[Any, List[float], float]:
    '''
    train for one epoch
    '''
    overhead = 0.0
    # check if the bit-width needs to be updated
    if epoch % assigner.ctx.assign_cycle == 1 and epoch != 1:
        if assigner.ctx.scheme in ['adaptive', 'random'] and engine.ctx.bit_type == BitType.QUANT:
            logger = logging.getLogger('trainer')
            logger.info(f'<epoch {epoch}, updating bit-width...>')
            ovearhead_start = time.time()
            bist_assignments = assigner.ctx.get_assignment(engine.ctx.send_idx)
            comm.ctx.update_buffer(bist_assignments)
            overhead = time.time() - ovearhead_start
        else:
            # uniform assignment do not need re-assignment
            pass
    # record epoch training time
    epoch_start = time.time()
    # forawrd
    model.train()
    logits = model(graph, input_data)
    loss = criterion(logits[train_mask], labels[train_mask]) / total_num_training_samples
    # backward
    optimizer.zero_grad()
    loss.backward()
    # update
    update_start = time.time()
    average_gradients(model)
    reduce_time = time.time() - update_start
    optimizer.step()
    epoch_time = time.time() - epoch_start
    # get time breakdown records
    traced_time = engine.ctx.timer.epoch_traced_time()
    engine.ctx.timer.clear()
    traced_time.insert(0, epoch_time)
    return overhead, loss, traced_time, reduce_time

@torch.no_grad()
def val_test(graph: DGLHeteroGraph, model: nn.Module, input_data: Tensor, labels: Tensor, train_mask: Tensor, val_mask: Tensor, test_mask: Tensor, is_multilabel: bool = False):
    '''
    perform validation / test
    '''
    # inference
    model.eval()
    logits = model(graph, input_data)
    metrics = []
    metrics.extend(get_metrics(labels[train_mask], logits[train_mask], is_multilabel))
    metrics.extend(get_metrics(labels[val_mask], logits[val_mask], is_multilabel))
    metrics.extend(get_metrics(labels[test_mask], logits[test_mask], is_multilabel))
    engine.ctx.timer.clear(is_train=False)
    return metrics

'''
*************************************************
*************** metric functions ****************
*************************************************
'''

def get_metrics(labels: Tensor, logits: Tensor, is_F1):
    '''
    get metrics for evaluation, use F1-score for multilabel classification tasks and accuracy for single label classification tasks
    '''
    if is_F1:
        # prepare for F1-score
        y_pred = (logits > 0)
        # get TP FP FN
        TP = torch.logical_and(y_pred == 1, labels == 1).float().sum()
        FP = torch.logical_and(y_pred == 1, labels == 0).float().sum()
        FN = torch.logical_and(y_pred == 0, labels == 1).float().sum()
        return [TP, TP + FP, TP + FN]
    else:
        # prepare for accuracy
        y_pred = torch.argmax(logits, dim = -1)
        is_label_correct = (y_pred == labels).float().sum()
        return [is_label_correct, labels.shape[0]]

def aggregate_accuracy(loss: Tensor, metrics: List[Union[float, int]], epoch: int) -> str:
    '''
    aggregate metrics from each worker for evaluation
    '''
    metrics = torch.FloatTensor(metrics)
    comm.all_reduce_sum(metrics)
    (train_acc, val_acc, test_acc) =  \
    (metrics[0] / metrics[1],
     metrics[2] / metrics[3],
     metrics[4] / metrics[5])
    comm.all_reduce_sum(loss)
    epoch_metrics = [train_acc, val_acc, test_acc]
    engine.ctx.recorder.add_new_metrics(epoch, epoch_metrics)
    return f'Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc * 100:.2f}% | Val Acc {val_acc * 100:.2f}% | Test Acc {test_acc * 100:.2f}%'

def aggregate_F1(loss: Tensor, metrics: List[Union[float, int]], epoch: int) -> str:
    '''
    aggregate metrics from each worker for evaluation
    '''
    def _safe_divide(numerator, denominator):
        if denominator == 0:
            denominator = 1
        return numerator / denominator
    comm.all_reduce_sum(metrics)
    # calculate precision and recall
    (train_precision, val_precision, test_precision) = \
        (_safe_divide(metrics[0], metrics[1]),
         _safe_divide(metrics[3], metrics[4]),
         _safe_divide(metrics[6], metrics[7]))
    (train_recall, val_recall, test_recall) = \
        (_safe_divide(metrics[0], metrics[2]),
         _safe_divide(metrics[3], metrics[5]),
         _safe_divide(metrics[6], metrics[8]))
    comm.all_reduce_sum(loss)
    train_f1_micro = _safe_divide(2 * train_precision * train_recall, train_precision + train_recall)
    val_f1_micro = _safe_divide(2 * val_precision * val_recall, val_precision + val_recall)
    test_f1_micro = _safe_divide(2 * test_precision * test_recall, test_precision + test_recall)
    epoch_metrics = [train_f1_micro, val_f1_micro, test_f1_micro]
    engine.ctx.recorder.add_new_metrics(epoch, epoch_metrics)
    return f'Epoch {epoch:05d} | Loss {loss.item():.4f} | Train F1 {train_f1_micro * 100:.2f} | Val F1 {val_f1_micro * 100:.2f} | Test F1 {test_f1_micro * 100:.2f}'

    
    
    


