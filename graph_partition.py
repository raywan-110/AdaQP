import argparse

from AdaQP.helper.partition import graph_patition_store

def main():
    parser = argparse.ArgumentParser(description='graph partition scripts')
    parser.add_argument('--dataset', type=str, default='reddit', help='training dataset')
    parser.add_argument('--raw_dir', type=str, default='data/dataset')
    parser.add_argument('--partition_dir', type=str, default='data/part_data')
    parser.add_argument('--partition_size', type=int, default=2)
    args = parser.parse_args()
    args = vars(args)
    # perform graph partition
    graph_patition_store(args['dataset'], args['partition_size'], args['raw_dir'], args['partition_dir'])
    graph_name = args['dataset']
    print(f'<{graph_name} graph partition done.>')


if __name__ == '__main__':
    main()