import time
import logging
import torch
from typing import Any, List, Union


class Recorder(object):
    def __init__(self, epoches: int):
        self.epoches_metrics = torch.zeros(epoches, 3)  # store each epoch's train/val/test metrics

    def add_new_metrics(self, epoch_count: int, epoch_metrics: List[Union[float, Any]]):
        '''
        epoch count is from 1 to epoches
        '''
        assert len(epoch_metrics) == 3
        self.epoches_metrics[epoch_count - 1] = torch.tensor(epoch_metrics) # [train, val, test] for each row

    def display_final_statistics(self, metrics_file:str = None, val_metric_curve_file: str = None, model_name='gcn'):
        '''
        display  and store statistics information about train/val/test metrics
        '''
        result = 100 * self.epoches_metrics
        argmax = result[:, 1].argmax().item()  # use valid metrics to find the best
        display_info = f'Highest Train: {result[:, 0].max():.2f}\n' + f'Highest Valid: {result[:, 1].max():.2f}\n' + f'  Final Train: {result[argmax, 0]:.2f}\n' + f'  Final Valid: {result[argmax, 1]:.2f}\n' + f'   Final Test: {result[argmax, 2]:.2f}'
        logging.info(display_info)
        # write results to file
        if metrics_file is not None:
            with open(metrics_file, 'a') as f:
                f.write(f'{model_name} runs on {time.strftime("%Y-%m-%d", time.localtime())}:\n')
                f.write(f'Highest Train: {result[:, 0].max():.2f}\n')
                f.write(f'Highest Valid: {result[:, 1].max():.2f}\n')
                f.write(f'  Final Train: {result[argmax, 0]:.2f}\n')
                f.write(f'  Final Valid: {result[argmax, 1]:.2f}\n')
                f.write(f'   Final Test: {result[argmax, 2]:.2f}\n')
        # write val metric curve to file
        if val_metric_curve_file is not None:
            val_metrics = result[:, 1]
            torch.save(val_metrics, val_metric_curve_file)
