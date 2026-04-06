import sys
import json
import os
import math
import yaml
import torch
import dgl
import random
import copy as cp
import numpy as np
import logging
from util import *
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from .dataset import TrainDataset
from .Env import Env

class GCATEnv(Env):
    def __init__(self, args):
        super(GCATEnv, self).__init__(args)
        
        self.nov_reward_map = self.load_nov_reward()
        self.concept_importance = self.load_concept_importance()
        
        if args.target_concepts != [0]:
          self.target_concepts = set(args.target_concepts)
        else:
          self.target_concepts = set(concept for concepts in self.know_map.values() for concept in concepts)

    def load_concept_importance(self):
        path = f'graph_data/{self.args.data_name}/K_Directed.txt'
        degrees = defaultdict(int)
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        u, v = int(parts[0]), int(parts[1])
                        degrees[u] += 1
                        degrees[v] += 1
        except FileNotFoundError:
            # If no graph file, assume uniform importance
            return defaultdict(lambda: 1.0)

        # Normalize
        max_deg = max(degrees.values()) if degrees else 1
        importance = {k: v / max_deg for k, v in degrees.items()}
        return importance

    def reset_with_users(self, uids):
        self.state = {}
        self.avail_questions = {}
        self.used_questions = {}
        self.know_stat = {}
        self.concept_consistency = {}

        for uu in uids:
            self.avail_questions[uu] = {
                qid for qid in self.sup_rates[uu].keys() if any(concept in self.target_concepts for concept in self.know_map[qid])
            }
            self.used_questions[uu] = []
            self.concept_consistency[uu] = {concept: False for concept in self.target_concepts}
        
        self.uids = uids
        self.cnt_step = 0
        self.state['batch_question']= np.zeros((len(uids), len(self.target_concepts)*2))
        self.state['batch_answer']= np.zeros((len(uids), len(self.target_concepts)*2))
        self.last_div = 0

        # Pass concept_map to get uncertainty if available
        result = self.model.get_knowledge_status(torch.tensor(self.uids).to(self.device), self.know_map)

        if isinstance(result, tuple):
            self.know_stat, self.uncertainty = result
        else:
            self.know_stat = result
            self.uncertainty = None

        self.previous_know_stat = self.know_stat * 0
        
        _, pred, correct_query = self.model.cal_loss(self.uids, self.query_rates, self.know_map)
        self.last_accuracy = np.zeros(len(uids))
        for i in range(len(pred)):
            pred_bin = np.where(pred[i] > 0.5, 1, 0)
            ACC = np.sum(np.equal(pred_bin, correct_query[i])) / len(pred_bin) 
            self.last_accuracy[i] = ACC

        return self.state

    def step(self, action, last_epoch):
        for i, uu in enumerate(self.uids):
            if action[i] not in self.avail_questions[uu]:
                self.logger.debug("action exited")
                pass

        reward, pred, label, rate = self.reward(action)
        
        self.previous_know_stat = self.know_stat

        result = self.model.get_knowledge_status(torch.tensor(self.uids).to(self.device), self.know_map)
        if isinstance(result, tuple):
            self.know_stat, self.uncertainty = result
        else:
            self.know_stat = result
            self.uncertainty = None
        
        user_dones = []

        coverages = []
        for i, uu in enumerate(self.uids):
            q = int(action[i])
            # If action is not available for this user, skip modification and compute coverage only
            if q not in self.avail_questions.get(uu, set()):
                self.logger.debug(f"action {q} not available for user {uu}; skipping update")
                # compute coverage without modifying avail/used
                all_concepts = set()
                tested_concepts = set()
                user_rates = self.rates.get(int(uu), {})
                for qid in user_rates:
                    all_concepts.update(set(self.know_map[qid]))
                used_qs = self.used_questions.get(int(uu), self.used_questions.get(uu, []))
                for used_qid in used_qs:
                    tested_concepts.update(set(self.know_map[used_qid]))

                if len(all_concepts) > 0:
                    cov = len(tested_concepts) / len(all_concepts)
                else:
                    cov = 0.0

                coverages.append(cov)

                all_stable = all(
                    self.concept_consistency.get(uu, self.concept_consistency.get(int(uu), {})).get(concept, False)
                    for concept in self.target_concepts
                )

                if (cov == 1.0 and all_stable) or len(self.avail_questions.get(uu, [])) < 1:
                    user_dones.append(True)
                else:
                    user_dones.append(False)
                continue

            # valid action: remove and record
            self.avail_questions[uu].remove(q)
            self.used_questions[uu].append(q)

            for concept in self.know_map[q]:
                if concept in self.target_concepts:

                  is_stable = False

                  if self.uncertainty is not None:
                      # Uncertainty-Based Termination
                      # self.uncertainty is (batch_size, num_concepts) variance
                      unc_val = self.uncertainty[i][concept]
                      # Threshold for variance. If variance is low, we are certain.
                      # Max variance of a probability p is 0.25 (at p=0.5).
                      # Let's say threshold is 0.01 (std dev 0.1).
                      if unc_val < 0.01:
                          is_stable = True
                          # print(f"Concept {concept} stable by uncertainty: {unc_val.item()}")
                  else:
                      # Fallback to delta
                      prev_score = self.previous_know_stat[i][concept]
                      curr_score = self.know_stat[i][concept]
                      error = abs(curr_score - prev_score)
                      if error < 0.010:
                          is_stable = True

                  if is_stable:
                      self.concept_consistency[uu][concept] = True
                      stable_questions = []
                      for qid in self.avail_questions[uu]:
                          concepts_in_question = self.know_map[qid]
                          all_concepts_stable = all(
                              concept in self.concept_consistency[uu] and self.concept_consistency[uu][concept]
                              for concept in concepts_in_question
                          )
                          
                          if all_concepts_stable:
                              stable_questions.append(qid)

                      for qid in stable_questions:
                          self.avail_questions[uu].remove(qid)

            all_concepts = set()
            tested_concepts = set()
            for qid in self.rates[uu]: 
                all_concepts.update(set(self.know_map[qid]))
            for qid in self.used_questions[uu]:
                tested_concepts.update(set(self.know_map[qid]))

            if len(all_concepts) > 0:
                coverage = len(tested_concepts) / len(all_concepts)
            else:
                coverage = 0.0
            coverages.append(coverage)

            all_stable = all(
                self.concept_consistency[uu][concept] 
                for concept in self.target_concepts
            )

            if (coverage == 1.0 and all_stable) or len(self.avail_questions[uu])<1:
                user_dones.append(True)
            else:
                user_dones.append(False)
              
        done = all(user_dones)
        cov = sum(coverages) / len(coverages)

        # Only include entries that have non-empty prediction/label arrays
        all_info = []
        for i, uu in enumerate(self.uids):
            p = pred[i] if i < len(pred) else []
            l = label[i] if i < len(label) else []
            rt = rate[i] if i < len(rate) else 0
            if hasattr(p, '__len__') and len(p) > 0 and hasattr(l, '__len__') and len(l) > 0:
                all_info.append({"pred": p, "label": l, "rate": rt})

        self.cnt_step += 1

        self.state['batch_question'][:, self.cnt_step] = action
        self.state['batch_answer'][:, self.cnt_step] = np.array(rate)+1 # idx=0 is pad

        return self.state, reward, done, all_info, coverages

    def reward(self, action):
        # update cdm
        records = []
        for i, uu in enumerate(self.uids):
            qid = int(action[i])
            # Skip update if user is finished (dummy action)
            if len(self.avail_questions.get(uu, [])) == 0:
                continue

            user_key = int(uu)
            rate_val = self.rates.get(user_key, {}).get(qid, 0)
            records.append((user_key, qid, rate_val))
        self.dataset = TrainDataset(records, self.know_map, self.user_num, self.item_num, self.know_num)
        self.model.update(self.dataset, self.args.cdm_lr, epochs=self.args.cdm_epoch, batch_size=self.args.cdm_bs)

        # eval on query
        loss, pred, correct_query = self.model.cal_loss(self.uids, self.query_rates, self.know_map)
        final_rate = [self.rates.get(int(uu), {}).get(int(action[i]), 0) for i, uu in enumerate(self.uids)]

        new_accuracy = np.zeros(len(self.uids))
        for i in range(len(pred)):
            pred_bin = np.where(pred[i] > 0.5, 1, 0)
            ACC = np.sum(np.equal(pred_bin, correct_query[i])) / len(pred_bin) 
            new_accuracy[i] = ACC

        acc_rwd = new_accuracy - self.last_accuracy
        self.last_accuracy = new_accuracy

        div_rwd = []
        for i, uu in enumerate(self.uids):
             # Calculate current coverage for user uu
             all_concepts = set()
             tested_concepts = set()
             user_key = int(uu)
             user_rates = self.rates.get(user_key, {})
             for qid in user_rates:
                 all_concepts.update(set(self.know_map[qid]))

             used_qs = self.used_questions.get(user_key, self.used_questions.get(uu, []))
             for used_qid in used_qs:
                 tested_concepts.update(set(self.know_map[used_qid]))

             if len(all_concepts) > 0:
                 cov = len(tested_concepts) / len(all_concepts)
             else:
                 cov = 0.0

             sup_keys = list(self.sup_rates.get(user_key, {}).keys())
             used_qs_for_call = used_qs
             conc_consistency = self.concept_consistency.get(user_key, self.concept_consistency.get(uu, {}))
             r = self.compute_div_reward(sup_keys, self.know_map, used_qs_for_call, int(action[i]), conc_consistency, cov)
             div_rwd.append(r)
        div_rwd = np.array(div_rwd)

        nov_rwd = np.array([self.nov_reward_map.get(int(action[i]), 0.0) for i in range(len(action))])

        rwd = np.concatenate((acc_rwd.reshape(-1,1), div_rwd.reshape(-1,1), nov_rwd.reshape(-1,1)),axis=-1) # (B,3)
        
        return rwd, pred, correct_query, final_rate
    
    def load_nov_reward(self):
        # use data dir discovered by Env.load_data()
        data_dir = getattr(self, 'data_dir', os.path.join(os.getcwd(), 'data'))
        path = os.path.join(data_dir, f'nov_reward_{self.args.data_name}.json')
        with open(path, encoding='utf8') as i_f:
            concept_data = json.load(i_f)
        
        nov_reward_map = {}
        for k,v in concept_data.items():
            qid_pad = int(k)
            nov_reward_map[qid_pad] = v
            
        return nov_reward_map
    
    def compute_div_reward(self, all_questions, concept_map, tested_questions, qid, concept_consistency, coverage=0.0):
        concept_cnt = set()
        
        for q in list(tested_questions):
            for c in concept_map[q]:
                concept_cnt.add(c)
        
        reward = 0.0

        # Calculate coverage deficit
        deficit = 1.0 - coverage

        for c in concept_map[qid]:
            imp = self.concept_importance.get(c, 0.5) # Default 0.5
            if c not in concept_cnt:
                # New concept
                reward += imp * 5.0 # Boost reward for new concepts to ensure high coverage
            elif not concept_consistency.get(c, False):
                # Unstable concept
                reward += imp

        # Scale by deficit (boost reward if coverage is low)
        return reward * (1.0 + deficit)
