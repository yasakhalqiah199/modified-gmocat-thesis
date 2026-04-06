import os
import sys
import time
import logging
import numpy as np
import dgl
import torch
from datetime import datetime

import envs as all_envs
import agents as all_agents
import function as all_FA
from util import get_objects, set_global_seeds, arg_parser, check_path

def str2bool(str=""):
    str = str.lower()
    if str.__contains__("yes") or str.__contains__("true") or str.__contains__("y") or str.__contains__("t"):
        return True
    else:
        return False

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run.py.
    """
    parser = arg_parser()
    parser.add_argument('-seed', type=int, default=145)
    parser.add_argument('-environment', type=str, default="GCATEnv")
    parser.add_argument('-data_path', type=str, default="./data/")
    parser.add_argument('-data_name', type=str, default="dbekt22")
    parser.add_argument('-agent', type=str, default="GCATAgent")
    parser.add_argument('-FA', type=str, default="GCAT")
    parser.add_argument('-CDM', type=str, default='NCD', help="type of CDM")
    parser.add_argument('-T', type=int, default=20, help="time_step")
    parser.add_argument('-ST', type=eval, default="[5, 10, 20]", help="evaluation_time_step")
    parser.add_argument('-gpu_no', type=str, default="0", help='which gpu for usage')
    parser.add_argument('-device', type=str, default='cpu', help='device to run on: cpu or cuda')
    
    parser.add_argument('-learning_rate', type=float, default=0.01, help="learning rate")
    parser.add_argument('-training_epoch', type=int, default=5, help="training epoch")
    parser.add_argument('-cdm_lr', type=float, default=0.01, help="cdm lr")
    parser.add_argument('-cdm_epoch',  type=int, default=5, help="cdm epoch")
    parser.add_argument('-cdm_bs',  type=int, default=128, help="cdm bs")
    
    parser.add_argument('-train_bs', type=int, default=50)
    parser.add_argument('-test_bs', type=int, default=50)
    parser.add_argument('-batch',  type=int, default=128, help="batch_size")
    
    parser.add_argument('-gamma', type=float, default=0.9, help="gamma")
    parser.add_argument('-latent_factor', type=int, default=256, help="latent factor")
    parser.add_argument('-n_block', type=int, default=2, help="")
    parser.add_argument('-graph_block', type=int, default=2, help="")
    parser.add_argument('-n_head', type=int, default=1, help="")
    parser.add_argument('-dropout_rate', type=float, default=0.0, help="")
    parser.add_argument('-policy_epoch', type=int, default=4, help="policy_epoch")
    parser.add_argument('-morl_weights', type=eval, default="[1,1,1]", help="")
    parser.add_argument('-emb_dim',  type=int, default=128, help="emb_dim")
    parser.add_argument('-use_graph', type=str2bool, default="True", help="")
    parser.add_argument('-use_attention', type=str2bool, default="True", help="")
    parser.add_argument('-store_action', type=str2bool, default="False", help="")
    parser.add_argument('-student_ids', type=eval, default="[0]")
    parser.add_argument('-target_concepts', type=eval, default="[0]") 
    return parser

def build_graph(type, node, path):
    g=dgl.graph(([],[]))
    g.add_nodes(node)
    edge_list = []
    # resolve provided path against several likely base directories
    base = path
    if not os.path.isdir(base):
        # try relative to this script
        candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), base))
        if os.path.isdir(candidate):
            base = candidate
        else:
            # try repo-level GMOCAT-Final/.. locations
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            candidate = os.path.join(repo_root, base)
            if os.path.isdir(candidate):
                base = candidate
    # determine file name for edge list
    fname = None
    if type == 'direct':
        fname = 'K_Directed.txt'
    elif type == 'k_from_e':
        fname = 'k_from_e.txt'
    elif type == 'e_from_k':
        fname = 'e_from_k.txt'

    # ensure node count is at least max index in file + 1
    try:
        max_idx = -1
        with open(os.path.join(base, fname), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    a, b = int(parts[0]), int(parts[1])
                    max_idx = max(max_idx, a, b)
        if max_idx >= 0 and node <= max_idx:
            node = max_idx + 1
    except FileNotFoundError:
        pass

    if type == 'direct':
        with open(os.path.join(base, 'K_Directed.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'k_from_e':
        with open(os.path.join(base, 'k_from_e.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g
    elif type == 'e_from_k':
        with open(os.path.join(base, 'e_from_k.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '').split('\t')
                edge_list.append((int(line[0]), int(line[1])))
        # add edges two lists of nodes: src and dst
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        return g

def construct_local_map(args, path):
    local_map = {
        'directed_g': build_graph('direct', args.know_num, path),
        'k_from_e': build_graph('k_from_e', args.know_num + args.item_num, path),
        'e_from_k': build_graph('e_from_k', args.know_num + args.item_num,path),
    }
    return local_map

def main(args):
    # arguments
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    
    # Auto-detect device if not specified or available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    args.model = "_".join([args.agent, args.FA, str(args.T)])
    # initialization
    set_global_seeds(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

    # logger
    logger = logging.getLogger(f'{args.FA}')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    
    # Determine project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'baseline_log', args.data_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(os.path.join(log_dir, f'{args.FA}_{args.data_name}_{args.CDM}_' + time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())) + '.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Training Model: "+args.model)

    # environments
    envs = get_objects(all_envs)
    env = envs[args.environment](args)
    # policy network
    args.user_num = env.user_num
    args.item_num = env.item_num
    args.know_num = env.know_num
    # print args
    logger.info("Hype-Parameters: "+str(args))
    print(args)
    # constuct graph

    local_map = construct_local_map(args, path=os.path.join(script_dir, 'graph_data', args.data_name) + os.sep)
    # local_map = None
    nets = get_objects(all_FA)
    fa = nets[args.FA].create_model(args, local_map)
    # agents
    agents = get_objects(all_agents)
    agents[args.agent](env, fa, args).train()




if __name__ == '__main__':
    main(sys.argv[1:])