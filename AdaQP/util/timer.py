import time
import torch
import os
import torch.distributed as dist
import numpy as np
from contextlib import contextmanager


class Timer(object):

    def __init__(self):
        super(Timer, self).__init__()
        self._record = {}
        self._total_record = []
        self.device = None

    @contextmanager
    def record(self, name):
        if name in self._record:
            raise Exception(f'{name} already exists')
        torch.cuda.current_stream(self.device).synchronize()
        start = time.time()
        yield
        torch.cuda.current_stream(self.device).synchronize()
        end = time.time()
        self._record[name] = (start, end)

    def epoch_time(self):
        total_com = 0
        total_quant = 0
        total_dequant = 0
        total_central_agg = 0
        total_marginal_agg = 0
        total_full_agg = 0
        for name, (start, end) in self._record.items():
            if 'communication' in name:
                total_com += end - start
            elif 'quantization' in name and 'de' not in name:
                total_quant += end - start
            elif 'de-quantization' in name:
                total_dequant += end - start
            elif 'central' in name:
                total_central_agg += end - start
            elif 'marginal' in name:
                total_marginal_agg += end - start
            elif 'full' in name:
                total_full_agg += end - start
            else:
                raise KeyError(f'no {name} key')
        return [total_com, total_quant + total_dequant, total_central_agg, total_marginal_agg, total_full_agg]

    def clear(self, is_train=True):
        # store in the _total_record for backup
        if is_train:
            self._total_record.append(self.epoch_time())
        self._record = {}

    def persist(self, run, bits):
        dir = 'time_record'
        if not os.path.exists(dir):
            os.mkdir(dir)
        np.save(file=f'{dir}/run{run}_{bits}bits_worker{dist.get_rank()}_time_record.npy', arr=self._total_record)
        self._total_record = []
