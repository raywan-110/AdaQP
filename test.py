import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='distributed full graph GCN training on obgn-arxiv')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', help='training dataset')
    parser.add_argument('--raw_dir', type=str, default='/data/borui/dataset')
    parser.add_argument('--partition_dir', type=str, default='/data/borui/part_data')
    parser.add_argument('--exp_dir', type=str, default='exp')
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='gcn')
    
    p_args = parser.parse_args()
    runtime_args = vars(p_args)
    print(runtime_args)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    print(dir_path)

if __name__ == '__main__':
    main()