import numpy as np
import time
import logging
import os
import json
import sys
import random
import torch
from datetime import datetime
from util import set_global_seeds, arg_parser
from envs.dataset import TrainDataset 
from envs.irt import IRTModel
from envs.ncd import NCDModel

def common_arg_parser():
    parser = arg_parser()
    parser.add_argument('-seed', type=int, default=145)
    parser.add_argument('-data_name', type=str, default='dbekt22')
    parser.add_argument('-CDM', dest='CDM', type=str, default='NCD', help="type of CDM")
    parser.add_argument('-T', dest='T', type=int, default=20, help="time_step")

    parser.add_argument('-gpu_no', dest='gpu_no', type=str, default="0", help='which gpu for usage')
    parser.add_argument('-device', type=str, default='cpu', help='device to run on: cpu or cuda')
    
    parser.add_argument('-learning_rate', dest='learning_rate', type=float, default=0.002, help="learning rate")
    parser.add_argument('-training_epoch', dest='training_epoch', type=int, default=100, help="training epoch")
    parser.add_argument('-batch_size', dest='batch_size', type=int, default=2048, help="batch_size")
    
    return parser

def main(args):
    # arguments
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)

    # Auto-detect device if not specified or available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    args.path = "_".join([args.CDM, args.data_name, str(args.T)])

    # initialization
    set_global_seeds(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

    # Determine project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # logger
    logger = logging.getLogger("Pretrain")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    log_dir = os.path.join(script_dir, 'pretrain_log', args.data_name)
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{args.data_name}_{args.CDM}_' + time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())) + '.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pre Training CDM: " + args.path)
    
    data_dir = os.path.join(script_dir, 'data')
    with open(os.path.join(data_dir, f'concept_map_{args.data_name}.json'), encoding='utf8') as i_f:
        concept_data = json.load(i_f) 
    with open(os.path.join(data_dir, f'train_task_{args.data_name}.json'), encoding='utf8') as i_f:
        stus = json.load(i_f) 

    rates = {}
    items = set()
    user_cnt = -1
    know_map = {}
    for stu in stus:
        user_cnt += 1
        rates[user_cnt] = {}
        for qid, label in zip(stu['q_ids'], stu['labels']): 
            rates[user_cnt][int(qid)+1] = int(label)
            items.add(int(qid)+1)
            know_map[int(qid)+1] = concept_data[str(qid)]

    max_itemid = max(items)

    knows = set()
    for know_list in know_map.values():
        knows.update(know_list)
  
    max_knowid = max(knows)

    user_num = len(rates)
    print('user num', user_num, user_cnt+1)

    item_num = max_itemid + 1
    know_num = max_knowid + 1

    logger.info(f"user num: {user_num}, item num: {item_num}, know_num: {know_num}")
 
    N = user_num//10
    test_fold, valid_fold = 0, 1
    all_users = [idx for idx in range(user_num)]
    random.Random(args.seed).shuffle(all_users)
    
    train_users = [uu for i, uu in enumerate(all_users) if i //
                    N != test_fold and i//N != valid_fold ]

    print('use user num: ', len(train_users))
    logger.info(f'use user num: {len(train_users)}')
    records = []
    for uu in train_users:
        u_record = [(uu, pair[0], pair[1]) for pair in list(rates[uu].items())]
        records.extend(u_record)

    dataset = TrainDataset(records, know_map, user_num, item_num, know_num)
    
    name = args.CDM
    if name == 'IRT':
        cdm = IRTModel(args, user_num, item_num, 1)
    elif name == 'NCD':
        cdm = NCDModel(args, user_num, item_num, know_num)
    else:
        logger.info("CDM no exist")
        exit()

    print(args)
    logger.info("Hype-Parameters: " + str(args))
   
    model_dir = os.path.join(script_dir, 'models', args.data_name)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f'{args.path}.pt')
    cdm.train(dataset, args.learning_rate, args.batch_size, epochs=args.training_epoch,path=path)


if __name__ == '__main__':
    main(sys.argv[1:])