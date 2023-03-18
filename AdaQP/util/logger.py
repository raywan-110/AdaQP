import torch
import numpy as np
import time


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, rank, run=None, result_file=None, model='gcn'):
        if rank == 0:
            if run is not None:
                result = 100 * torch.tensor(np.array(self.results[run]))
                argmax = result[:, 1].argmax().item()  # use valid result to find the best
                print(f'Run {run + 1:02d}:')
                print(f'Highest Train: {result[:, 0].max():.2f}')
                print(f'Highest Valid: {result[:, 1].max():.2f}')
                print(f'  Final Train: {result[argmax, 0]:.2f}')
                print(f'   Final Test: {result[argmax, 2]:.2f}')
            else:
                result = 100 * torch.tensor(np.array(self.results))

                best_results = []
                for r in result:
                    train1 = r[:, 0].max().item()
                    valid = r[:, 1].max().item()
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test = r[r[:, 1].argmax(), 2].item()
                    best_results.append((train1, valid, train2, test))

                best_result = torch.tensor(best_results)

                print('All runs:')
                r = best_result[:, 0]
                print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 1]
                print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 2]
                print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
                r = best_result[:, 3]
                print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

                # write results to file
                if result_file is not None:
                    with open(result_file, 'a') as f:
                        f.write(f'{model} all runs on {time.strftime("%Y-%m-%d", time.localtime())}:\n')
                        r = best_result[:, 0]
                        f.write(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}\n')
                        r = best_result[:, 1]
                        f.write(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}\n')
                        r = best_result[:, 2]
                        f.write(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}\n')
                        r = best_result[:, 3]
                        f.write(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}\n')
        else:
            pass
