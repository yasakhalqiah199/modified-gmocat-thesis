import json
import numpy as np
from collections import Counter
import json
import os
import argparse

def dump_json(path_, data):
    with open(path_, 'w') as fh:
        json.dump(data, fh, indent=2)
    return data

def build_local_map(name, data_dir, graph_dir):
    concept_file = os.path.join(data_dir, f'concept_map_{name}.json')
    exer_n = 0
    if name == 'assist2009':
        exer_n = 17751+1
    elif name == 'junyi':
        exer_n = 2835+1
    elif name == '3_4':
        exer_n = 948+1
    else:
        # Dynamic calculation for other datasets like dbekt22
        q_map_file = os.path.join(data_dir, f'question_map_{name}.json')
        if os.path.exists(q_map_file):
            with open(q_map_file, 'r') as f:
                qmap = json.load(f)
            exer_n = len(qmap) + 1
        else:
            print(f"Warning: question_map_{name}.json not found, using default exer_n=0 which might be wrong.")

    temp_list = []
    with open(concept_file, encoding='utf8') as f:
        concept_map = json.load(f)
    k_from_e = '' # e(src) to k(dst)
    e_from_k = '' # k(src) to k(dst)
    
    for qid in concept_map:
        # has id=0 question for pad
        exer_id = int(qid) + 1
        for k in concept_map[str(qid)]:
            if (str(exer_id) + '\t' + str(k + exer_n)) not in temp_list or (str(k + exer_n) + '\t' + str(exer_id)) not in temp_list:
                k_from_e += str(exer_id) + '\t' + str(k + exer_n) + '\n'
                e_from_k += str(k + exer_n) + '\t' + str(exer_id) + '\n'
                temp_list.append((str(exer_id) + '\t' + str(k + exer_n)))
                temp_list.append((str(k + exer_n) + '\t' + str(exer_id)))

    path = os.path.join(graph_dir, name)
    if not os.path.exists(path):
        os.makedirs(path)

    with open(os.path.join(path, 'k_from_e.txt'), 'w') as f:
        f.write(k_from_e)
    with open(os.path.join(path, 'e_from_k.txt'), 'w') as f:
        f.write(e_from_k)

def constructDependencyMatrix(name, data_dir, graph_dir):
    data_file = os.path.join(data_dir, f'train_task_{name}.json')
    concept_file = os.path.join(data_dir, f'concept_map_{name}.json')
    
    knowledge_n = 0
    if name == 'assist2009':
        knowledge_n = 123 # num of knowledge
    elif name == 'junyi':
        knowledge_n = 40
    elif name == '3_4':
        knowledge_n = 86
    else:
        # Dynamic calculation
        with open(concept_file, encoding='utf8') as f:
            concept_map = json.load(f)
        max_k = 0
        for v in concept_map.values():
            if v:
                max_k = max(max_k, max(v))
        knowledge_n = max_k + 1

    edge_dic_deno = {}
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    
    with open(concept_file, encoding='utf8') as f:
        concept_map = json.load(f)
    # Calculate correct matrix
    knowledgeCorrect = np.zeros([knowledge_n, knowledge_n])
    for student in data:
        if student['log_num'] < 2:
            continue
        q_ids, labels = student['q_ids'], student['labels']
        for log_i in range(student['log_num']-1):
            if labels[log_i] * labels[log_i+1] == 1:
                # Check if q_ids are in concept_map (they are strings in json keys)
                qid1 = str(q_ids[log_i])
                qid2 = str(q_ids[log_i+1])
                if qid1 in concept_map and qid2 in concept_map:
                    for ki in concept_map[qid1]:
                        for kj in concept_map[qid2]:
                            if ki != kj:
                                # n_{ij}
                                knowledgeCorrect[ki][kj] += 1.0
                                # n_{i*}, calculate the number of correctly answering i
                                if ki in edge_dic_deno.keys():
                                    edge_dic_deno[ki] += 1
                                else:
                                    edge_dic_deno[ki] = 1

    s = 0
    c = 0
    # Calculate transition matrix
    knowledgeDirected = np.zeros([knowledge_n, knowledge_n])
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if i != j and knowledgeCorrect[i][j] > 0:
                    knowledgeDirected[i][j] = float(knowledgeCorrect[i][j]) / edge_dic_deno[i]
                    s += knowledgeDirected[i][j]
                    c += 1
    o = np.zeros([knowledge_n, knowledge_n])
    min_c = 100000000
    max_c = 0
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                min_c = min(min_c, knowledgeDirected[i][j])
                max_c = max(max_c, knowledgeDirected[i][j])
    s_o = 0
    l_o = 0
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if knowledgeCorrect[i][j] > 0 and i != j:
                o[i][j] = (knowledgeDirected[i][j] - min_c) / (max_c - min_c)
                l_o += 1
                s_o += o[i][j]
    
    # avg = 0.02
    if l_o > 0:
        avg = s_o / l_o #total / count
    else:
        avg = 0.0

    if name == 'assist2009':
        avg *= avg
        avg *= avg
    elif name == '3_4':
        # avg = s_o / l_o #total / count # 0.02
        # avg =0.02
        pass
    else:
        # Default behavior for new datasets
        # avg *= avg
        # avg *= avg
        pass
        
    print(f"Threshold avg: {avg}")
    # avg is threshold
    graph = ''
    # edge = np.zeros([knowledge_n, knowledge_n])
    for i in range(knowledge_n):
        for j in range(knowledge_n):
            if o[i][j] >= avg:
                graph += str(i) + '\t' + str(j) + '\n'
                # edge[i][j] = 1
    
    path = os.path.join(graph_dir, name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    with open(os.path.join(path, 'knowledgeGraph.txt'), 'w') as f:
        f.write(graph)

def process_edge(name, graph_dir):
    K_Directed = ''
    K_Undirected = ''
    edge = []
    path = os.path.join(graph_dir, name)
    
    with open(os.path.join(path, 'knowledgeGraph.txt'), 'r') as f:
        for i in f.readlines():
            i = i.replace('\n', '').split('\t')
            if len(i) >= 2:
                src = i[0]
                tar = i[1]
                edge.append((src, tar))
    visit = []
    for e in edge:
        if e not in visit:
            if (e[1],e[0]) in edge:
                K_Undirected += str(e[0] + '\t' + e[1] + '\n')
                visit.append(e)
                visit.append((e[1],e[0]))
            else:
                K_Directed += str(e[0] + '\t' + e[1] + '\n')
                visit.append(e)

    with open(os.path.join(path, 'K_Directed.txt'), 'w') as f:
        f.write(K_Directed)
    with open(os.path.join(path, 'K_Undirected.txt'), 'w') as f:
        f.write(K_Undirected)
    all = len(visit)
    print(f"Processed edges: {all}")

def nov_reward(dataset, data_dir):
    data_file = os.path.join(data_dir, f'train_task_{dataset}.json')
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)

    all_questions = []
    for student in data:
        q_ids = student['q_ids']
        all_questions.extend(q_ids)
    
    if not all_questions:
        print("No questions found in dataset.")
        return

    print(f"Min question ID: {min(all_questions)}")
    # Get Novel Items
    all_pairs = Counter(all_questions).items()
    item_freqs = [pair[1] for pair in all_pairs]

    threshold = np.quantile(item_freqs, q=0.9)
    print(f"Frequency threshold (90th percentile): {threshold}")
 
    less_popular_items = []
    for pair in all_pairs:
        if pair[1] <= threshold:
            less_popular_items.append(pair[0])
    print('number of less popular items is: ', len(less_popular_items))

    # Binary Novelty Reward System
    binary_nov_reward= {}
    for pair in all_pairs:
        if pair[1] <= threshold:
            binary_nov_reward[str(pair[0])] = 1
        else:
            binary_nov_reward[str(pair[0])] = 0
    
    dump_json(os.path.join(data_dir, f'nov_reward_{dataset}.json'), binary_nov_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dbekt22')
    args = parser.parse_args()
    
    name = args.dataset
    
    # Determine project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming standard structure: GMOCAT-modif/construct_graphs.py -> data is in GMOCAT-modif/data
    data_dir = os.path.join(script_dir, 'data')
    graph_dir = os.path.join(script_dir, 'graph_data')
    
    print(f"Processing dataset: {name}")
    print(f"Data directory: {data_dir}")
    print(f"Graph directory: {graph_dir}")
    
    build_local_map(name, data_dir, graph_dir)
    constructDependencyMatrix(name, data_dir, graph_dir)
    process_edge(name, graph_dir)
    nov_reward(name, data_dir)