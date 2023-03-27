import argparse

from AdaQP import Trainer

def main():
    parser = argparse.ArgumentParser(description='distributed full graph training')
    parser.add_argument('--dataset', type=str, default='reddit', help='training dataset')
    parser.add_argument('--num_parts', type=int, default=4, help='number of partitions')
    parser.add_argument('--backend', type=str, default='gloo',help='backend for distributed training')
    parser.add_argument('--init_method', type=str, default='env://',help='init method for distributed training')
    parser.add_argument('--model_name', type=str, default='gcn', help='model for training')
    parser.add_argument('--mode', type=str, default='AdaQP', help='running mode. optional modes: [Vanilla, AdaQP, AdaQP-q, AdaQP-p]')
    parser.add_argument('--assign_scheme', type=str, default='adaptive', help='bit-width assignment scheme. optional schemes: [adaptive, random, uniform]')
    parser.add_argument('--logger_level', type=str, default='INFO', help='logger level')
    parser.add_argument('--use_parallel', action='store_true')
    args = parser.parse_args()
    # init the trainer
    trainer = Trainer(args)
    # start training according to the config
    time_record = trainer.train()
    # store the training records
    trainer.save(time_record)

if __name__ == '__main__':
    main()
    
    