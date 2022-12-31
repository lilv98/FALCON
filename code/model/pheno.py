from email.policy import default
from multiprocessing.sharedctypes import Value
from tabnanny import verbose
import torch
import pdb
import argparse
import os
import pandas as pd
import numpy as np
import random
import tqdm
import re
import pickle
import torch.utils.checkpoint as checkpoint
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_rights(right):
    ridges = []
    axiom_parts = right.split(' ')
    counter = 0
    for i, part in enumerate(axiom_parts):
        counter += part.count('(')
        counter -= part.count(')')
        if counter == 0:
            ridges.append(i)
    if len(ridges) == 1:
        return [right]
    elif len(ridges) > 1:
        rights = []
        for i in range(len(ridges)):
            if i == 0:
                rights.append(' '.join(axiom_parts[: ridges[0] + 1]))
            else:
                rights.append(' '.join(axiom_parts[ridges[i - 1] + 1: ridges[i] + 1]))
        return rights
    else:
        raise ValueError

def get_all_concepts_and_relations(axiom, all_concepts, all_relations):
    
    if axiom[0] == '<' or axiom == 'owl:Thing':
        all_concepts.add(axiom)
        
    elif axiom == 'owl:Nothing':
        pass
    
    elif axiom[:21] == 'ObjectIntersectionOf(':
        axiom_parts = axiom.split(' ')
        counter = 0
        for i, part in enumerate(axiom_parts):
            counter += part.count('(')
            counter -= part.count(')')
            if counter == 1:
                ridge = i
                break
        left = ' '.join(axiom_parts[:ridge + 1])[21:]
        get_all_concepts_and_relations(left, all_concepts, all_relations)
        right = ' '.join(axiom_parts[ridge + 1:])[:-1]
        rights = get_rights(right)
        for every_right in rights:
            get_all_concepts_and_relations(every_right, all_concepts, all_relations)
        
    elif axiom[:14] == 'ObjectUnionOf(':
        axiom_parts = axiom.split(' ')
        counter = 0
        for i, part in enumerate(axiom_parts):
            counter += part.count('(')
            counter -= part.count(')')
            if counter == 1:
                ridge = i
                break
        left = ' '.join(axiom_parts[:ridge + 1])[14:]
        get_all_concepts_and_relations(left, all_concepts, all_relations)
        right = ' '.join(axiom_parts[ridge + 1:])[:-1]
        rights = get_rights(right)
        for every_right in rights:
            get_all_concepts_and_relations(every_right, all_concepts, all_relations)
        
    elif axiom[:22] == 'ObjectSomeValuesFrom(<':
        axiom_parts = axiom.split(' ')
        all_relations.add(axiom_parts[0][21:])
        right = ' '.join(axiom_parts[1:])[:-1]
        get_all_concepts_and_relations(right, all_concepts, all_relations)
        
    elif axiom[:21] == 'ObjectAllValuesFrom(<':
        axiom_parts = axiom.split(' ')
        all_relations.add(axiom_parts[0][20:])
        right = ' '.join(axiom_parts[1:])[:-1]
        get_all_concepts_and_relations(right, all_concepts, all_relations)
        
    elif axiom[:19] == 'ObjectComplementOf(':
        get_all_concepts_and_relations(axiom[19:-1], all_concepts, all_relations)
        
    else:
        raise ValueError

def extract_nodes(cfg, filename):
    outliers = ['ObjectOneOf', 
                'ObjectHasSelf', 
                'ObjectHasValue', 
                'ObjectMinCardinality', 
                'ObjectExactCardinality',
                'Data',
                ]
    tbox_name_str = []
    tbox_name_list = []
    tbox_desc = []
    with open(cfg.data_root + filename) as f:
        for line in f:
            line = line.strip('\n')
            flag = 0
            for outlier in outliers:
                if outlier in line:
                    flag = 1
                    break
            if flag == 0:
                name_match = re.match(r'ObjectIntersectionOf\(<(.*)> ObjectComplementOf\(<(.*)>\)\)', line, re.M|re.I)
                name_match_thing = re.match(r'ObjectIntersectionOf\(<(.*)> ObjectComplementOf\(owl:Thing\)\)', line, re.M|re.I)
                if name_match:
                    matched = name_match.groups()
                    tbox_name_list.append([f'<{matched[0]}>', 'subClassOf', f'<{matched[1]}>'])
                    tbox_name_str.append(line)
                elif name_match_thing:
                    matched = name_match_thing.groups()
                    tbox_name_list.append([f'<{matched[0]}>','subClassOf', f'owl:Thing'])
                    tbox_name_str.append(line)
                else:
                    tbox_desc.append(line)
    all_concepts = set()
    all_relations = set()
    for axiom in tbox_name_str:
        get_all_concepts_and_relations(axiom, all_concepts, all_relations)
    for axiom in tbox_desc:
        get_all_concepts_and_relations(axiom, all_concepts, all_relations)
    assert len(tbox_name_list) == len(tbox_name_str)
    return pd.DataFrame(tbox_name_list, columns=['h', 'r', 't']), tbox_desc, all_concepts, all_relations

def read_abox_ee(cfg):
    ret = []
    with open(cfg.data_root + 'MGI_Geno_DiseaseDO.rpt') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            allele_id = line[2].replace(':', '_')
            omim_id = line[-1].replace(':', '_')
            ret.append([allele_id, 'interactWith', omim_id])
    ret = pd.DataFrame(ret, columns=['h', 'r', 't']).drop_duplicates()
    return ret

def read_abox_ec(cfg):
    pheno_prefix = 'http://purl.obolibrary.org/obo/'
    ret = []
    alleles = set()
    with open(cfg.data_root + 'MGI_Geno_DiseaseDO.rpt') as f:
        for line in f:
            line = line.strip('\n').split('\t')
            allele_id = line[2].replace(':', '_')
            pheno_id = line[4].replace(':', '_')
            ret.append([allele_id, 'hasPhenotype', '<' + pheno_prefix + pheno_id + '>'])
            alleles.add(allele_id)
    omims = set()
    with open(cfg.data_root + 'phenotype.hpoa') as f:
        for line in f:
            if line[0] != '#':
                line = line.strip('\n').split('\t')
                omim_id = line[0].replace(':', '_')
                pheno_id = line[3].replace(':', '_')
                ret.append([omim_id, 'hasPhenotype', '<' + pheno_prefix + pheno_id + '>'])
                omims.add(omim_id)
    ret = pd.DataFrame(ret, columns=['h', 'r', 't']).drop_duplicates()
    return ret, alleles, omims

def get_abox_ec_created(all_concepts, k):
    ret = []
    counter = 0
    for concept in all_concepts:
        if counter < k:
            ret.append([concept[:-1] + '_1>', concept])
            counter += 1
    return pd.DataFrame(ret, columns=['h', 't'])

def get_data(cfg):
    tbox_name_train, tbox_desc_train, all_concepts, tbox_relations_train = extract_nodes(cfg, filename='pheno.txt')
    abox_ec, alleles, omims = read_abox_ec(cfg)
    abox_ee = read_abox_ee(cfg)

    abox_ec_created = get_abox_ec_created(all_concepts, k=cfg.n_abox_ec_created)
    created_entities = set(abox_ec_created.h.unique())

    # all_entities = set(abox_ec.h.unique()) | set(abox_ee.h.unique()) | set(abox_ee.t.unique())
    all_alleles = list(alleles | set(abox_ee.h.unique()))
    print(f'Alleles: {len(all_alleles)}')
    all_omims = list(omims | set(abox_ee.t.unique()))
    print(f'Omims: {len(all_omims)}')
    all_entities = [*all_alleles, *all_omims]
    all_relations = tbox_relations_train | set(['subClassOf', 'interactWith', 'hasPhenotype'])

    abox_ec = abox_ec[abox_ec.h.isin(all_entities)]
    abox_ec = abox_ec[abox_ec.t.isin(all_concepts)]
    abox_ee = abox_ee[abox_ee.h.isin(all_entities)]
    abox_ee = abox_ee[abox_ee.t.isin(all_entities)]

    c_dict = {k: v for v, k in enumerate(all_concepts)}
    e_dict = {k: v for v, k in enumerate(all_entities)}
    e_dict_created = {k: v + len(e_dict) for v, k in enumerate(created_entities)}
    e_dict_more = {**e_dict, **e_dict_created}
    r_dict = {k: v for v, k in enumerate(all_relations)}

    # id mapper
    tbox_name_train.h = tbox_name_train.h.map(c_dict)
    tbox_name_train.r = tbox_name_train.r.map(r_dict)
    tbox_name_train.t = tbox_name_train.t.map(c_dict)
    
    abox_ec.h = abox_ec.h.map(e_dict)
    abox_ec.r = abox_ec.r.map(r_dict)
    abox_ec.t = abox_ec.t.map(c_dict)
    
    abox_ec_created.h = abox_ec_created.h.map(e_dict_more)
    abox_ec_created.t = abox_ec_created.t.map(c_dict)

    abox_ee.h = abox_ee.h.map(e_dict)
    abox_ee.r = abox_ee.r.map(r_dict)
    abox_ee.t = abox_ee.t.map(e_dict)
    
    # abox splitter
    try:
        abox_ee_train = load_obj(cfg.data_root + 'abox_ee_train.pkl')
        abox_ee_test = load_obj(cfg.data_root + 'abox_ee_test.pkl')
    except:
        print('New Split!', flush=True)
        abox_ee_test = abox_ee[abox_ee['h'] > 2000].sample(frac=1)
        abox_ee_train = pd.concat([abox_ee, abox_ee_test]).drop_duplicates(keep=False)
        save_obj(abox_ee_train, cfg.data_root + 'abox_ee_train.pkl')
        save_obj(abox_ee_test, cfg.data_root + 'abox_ee_test.pkl')
    
    already_ts_dict = {}
    already_hs_dict = {}
    already_ts = abox_ee_train.groupby(['h', 'r'])['t'].apply(list).reset_index(name='ts').values
    already_hs = abox_ee_train.groupby(['t', 'r'])['h'].apply(list).reset_index(name='hs').values
    for record in already_ts:
        already_ts_dict[(record[0], record[1])] = record[2]
    for record in already_hs:
        already_hs_dict[(record[0], record[1])] = record[2]
    
    return tbox_name_train, tbox_desc_train, abox_ec, abox_ec_created, abox_ee_train, abox_ee_test, c_dict, e_dict, e_dict_more, r_dict, all_alleles, all_omims, already_ts_dict, already_hs_dict

class EEDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, e_dict, all_alleles, all_omims, already_ts_dict, already_hs_dict, stage):
        super().__init__()
        self.stage = stage
        self.cfg = cfg
        self.e_dict = e_dict
        self.all_alleles = all_alleles
        self.all_omims = all_omims
        self.data = torch.tensor(data.values)
        self.all_candidate = torch.arange(len(e_dict)).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng//2, 1)
        self.already_ts_dict = already_ts_dict
        self.already_hs_dict = already_hs_dict
        
    def sampling(self, pos):
        head, rel, tail = pos
        already_ts = torch.tensor(self.already_ts_dict[(head.item(), rel.item())])
        already_hs = torch.tensor(self.already_hs_dict[(tail.item(), rel.item())])
        neg_pool_t = torch.ones(len(self.all_omims))
        neg_pool_t[already_ts] = 0
        neg_pool_t = neg_pool_t.nonzero()
        neg_pool_h = torch.ones(len(self.all_alleles))
        neg_pool_h[already_hs] = 0
        neg_pool_h = neg_pool_h.nonzero()
        neg_t = neg_pool_t[torch.randint(len(neg_pool_t), (self.cfg.num_ng//2,))] + len(self.all_alleles)
        neg_h = neg_pool_h[torch.randint(len(neg_pool_h), (self.cfg.num_ng//2,))]
        return neg_t, neg_h
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.stage == 'train':
            pos = self.data[idx]
            neg_t, neg_h = self.sampling(pos)
            replace_tail = torch.cat([pos[0].expand_as(self.neg_shape), pos[1].expand_as(self.neg_shape), neg_t], dim=-1)
            replace_head = torch.cat([neg_h, pos[1].expand_as(self.neg_shape), pos[2].expand_as(self.neg_shape)], dim=-1)
            return torch.cat([pos.unsqueeze(dim=0), replace_tail, replace_head], dim=0)
        elif self.stage == 'test':
            pos = self.data[idx]
            replace_tail = torch.cat([pos[0].expand_as(self.all_candidate), pos[1].expand_as(self.all_candidate), self.all_candidate], dim=-1)
            replace_head = torch.cat([self.all_candidate, pos[1].expand_as(self.all_candidate), pos[2].expand_as(self.all_candidate)], dim=-1)
            return torch.cat([replace_head, replace_tail], dim=0), pos
        else:
            raise ValueError

class AboxECDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, e_dict):
        super().__init__()
        self.cfg = cfg
        self.e_dict = e_dict
        self.data = torch.tensor(data.values)
        self.all_candidate = torch.arange(len(e_dict)).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pos = self.data[idx]
        negs = self.all_candidate[torch.randint(len(self.e_dict), (self.cfg.num_ng,))]
        replace_head = torch.cat([negs, pos[1].expand_as(self.neg_shape), pos[2].expand_as(self.neg_shape)], dim=-1)
        return torch.cat([pos.unsqueeze(dim=0), replace_head], dim=0)

class AboxECCreatedDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data, e_dict):
        super().__init__()
        self.cfg = cfg
        self.e_dict = e_dict
        self.data = torch.tensor(data.values)
        self.all_candidate = torch.arange(len(e_dict)).unsqueeze(dim=-1)
        self.neg_shape = torch.zeros(self.cfg.num_ng, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pos = self.data[idx]
        negs = self.all_candidate[torch.randint(len(self.e_dict), (self.cfg.num_ng,))]
        replace_head = torch.cat([negs, pos[1].expand_as(self.neg_shape)], dim=-1)
        return torch.cat([pos.unsqueeze(dim=0), replace_head], dim=0)

class NaiveDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class FALCON(torch.nn.Module):
    def __init__(self, c_dict, e_dict, r_dict, cfg, device):
        super().__init__()
        self.c_dict = c_dict
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.n_entity = len(e_dict) + cfg.anon_e
        self.anon_e = cfg.anon_e
        self.cfg = cfg

        self.c_embedding = torch.nn.Embedding(len(c_dict), cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(len(r_dict), cfg.emb_dim)
        self.e_embedding = torch.nn.Embedding(len(e_dict), cfg.emb_dim)
        self.fc_0 = torch.nn.Linear(cfg.emb_dim * 2, 1)
        # self.fc_1 = torch.nn.Linear(cfg.emb_dim, 1)   
        
        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_0.weight.data)
        # torch.nn.init.xavier_uniform_(self.fc_1.weight.data) 
        
        self.max_measure = cfg.max_measure
        self.t_norm = cfg.t_norm
        self.nothing = torch.zeros(self.n_entity).to(device)
        self.residuum = cfg.residuum
        self.device = device
    
    def _logical_and(self, x, y):
        if self.t_norm == 'product':
            return x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).min(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return (((x + y -1) > 0) * (x + y - 1)).squeeze(dim=-2)
        else:
            raise ValueError

    def _logical_or(self, x, y):
        if self.t_norm == 'product':
            return x + y - x * y
        elif self.t_norm == 'minmax':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return torch.cat([x, y], dim=-2).max(dim=-2)[0]
        elif self.t_norm == 'Łukasiewicz':
            x = x.unsqueeze(dim=-2)
            y = y.expand_as(x)
            return 1 - ((((1-x) + (1-y) -1) > 0) * ((1-x) + (1-y) - 1)).squeeze(dim=-2)
        else:
            raise ValueError
    
    def _logical_not(self, x):
        return 1 - x
    
    def _logical_residuum(self, r_fs, c_fs):
        if self.residuum == 'notCorD':
            return self._logical_or(self._logical_not(r_fs), c_fs.unsqueeze(dim=-2))
        else:
            raise ValueError
    
    def _logical_exist(self, r_fs, c_fs):
        return self._logical_and(r_fs, c_fs).max(dim=-1)[0]

    def _logical_forall(self, r_fs, c_fs):
        return self._logical_residuum(r_fs, c_fs).min(dim=-1)[0]

    def _get_c_fs(self, c_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0)
        c_emb = c_emb.expand_as(e_emb)
        emb = torch.cat([c_emb, e_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def _get_c_fs_batch(self, c_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0).unsqueeze(dim=0).repeat(c_emb.size()[0], 1, 1)
        c_emb = c_emb.unsqueeze(dim=1).expand_as(e_emb)
        emb = torch.cat([c_emb, e_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def _get_r_fs(self, r_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0)
        l_emb = (e_emb + r_emb.unsqueeze(dim=0)).unsqueeze(dim=1).repeat(1, self.n_entity, 1)
        r_emb = e_emb.unsqueeze(dim=0).repeat(self.n_entity, 1, 1)
        emb = torch.cat([l_emb, r_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def _get_r_fs_batch(self, r_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0).unsqueeze(dim=0).repeat(r_emb.size()[0], 1, 1)
        l_emb = (e_emb + r_emb.unsqueeze(dim=1).expand_as(e_emb)).unsqueeze(dim=1).repeat(1, self.n_entity, 1, 1)
        r_emb = e_emb.unsqueeze(dim=2).repeat(1, 1, self.n_entity, 1)
        emb = torch.cat([l_emb, r_emb], dim=-1)
        # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)

    def forward(self, axiom, anon_e_emb):

        if axiom[0] == '<' or axiom == 'owl:Thing':
            c_id = torch.tensor(self.c_dict[axiom]).to(self.device)
            c_emb = self.c_embedding(c_id)
            # ret = self._get_c_fs(c_emb, anon_e_emb)
            ret = checkpoint.checkpoint(self._get_c_fs, c_emb, anon_e_emb)
            return ret
        
        elif axiom == 'owl:Nothing':
            return self.nothing
        
        elif axiom[:21] == 'ObjectIntersectionOf(':
            axiom_parts = axiom.split(' ')
            counter = 0
            for i, part in enumerate(axiom_parts):
                counter += part.count('(')
                counter -= part.count(')')
                if counter == 1:
                    ridge = i
                    break
            left = ' '.join(axiom_parts[:ridge + 1])[21:]
            right = ' '.join(axiom_parts[ridge + 1:])[:-1]
            rights = get_rights(right)
            ret = self._logical_and(self.forward(left, anon_e_emb), self.forward(rights[0], anon_e_emb))
            if len(rights) > 1:
                for every_right_id in range(1, len(rights)):
                    ret = self._logical_and(ret, self.forward(rights[every_right_id], anon_e_emb))
            return ret

        elif axiom[:14] == 'ObjectUnionOf(':
            axiom_parts = axiom.split(' ')
            counter = 0
            for i, part in enumerate(axiom_parts):
                counter += part.count('(')
                counter -= part.count(')')
                if counter == 1:
                    ridge = i
                    break
            left = ' '.join(axiom_parts[:ridge + 1])[14:]
            right = ' '.join(axiom_parts[ridge + 1:])[:-1]
            rights = get_rights(right)
            ret = self._logical_or(self.forward(left, anon_e_emb), self.forward(rights[0], anon_e_emb))
            if len(rights) > 1:
                for every_right_id in range(1, len(rights)):
                    ret = self._logical_or(ret, self.forward(rights[every_right_id], anon_e_emb))
            return ret

        elif axiom[:22] == 'ObjectSomeValuesFrom(<':
            axiom_parts = axiom.split(' ')
            r_id = torch.tensor(self.r_dict[axiom_parts[0][21:]]).to(self.device)
            r_emb = self.r_embedding(r_id)
            # r_fs = self._get_r_fs(r_emb, anon_e_emb)
            r_fs = checkpoint.checkpoint(self._get_r_fs, r_emb, anon_e_emb)
            right = ' '.join(axiom_parts[1:])[:-1]
            return self._logical_exist(r_fs, self.forward(right, anon_e_emb))

        elif axiom[:21] == 'ObjectAllValuesFrom(<':
            axiom_parts = axiom.split(' ')
            r_id = torch.tensor(self.r_dict[axiom_parts[0][20:]]).to(self.device)
            r_emb = self.r_embedding(r_id)
            # r_fs = self._get_r_fs(r_emb, anon_e_emb)
            r_fs = checkpoint.checkpoint(self._get_r_fs, r_emb, anon_e_emb)
            right = ' '.join(axiom_parts[1:])[:-1]
            return self._logical_forall(r_fs, self.forward(right, anon_e_emb))

        elif axiom[:19] == 'ObjectComplementOf(':
            return self._logical_not(self.forward(axiom[19:-1], anon_e_emb))

        else:
            raise ValueError

    def forward_name(self, x, anon_e_emb):
        c_emb_left = self.c_embedding(x[:, 0])
        c_emb_right = self.c_embedding(x[:, 2])
        fs_left = self._get_c_fs_batch(c_emb_left, anon_e_emb)
        fs_right = self._get_c_fs_batch(c_emb_right, anon_e_emb)
        return self.get_cc_loss(self._logical_and(fs_left, self._logical_not(fs_right))).mean()

    def get_cc_loss(self, fs):
        if self.max_measure == 'max':
            return - torch.log(1 - fs.max(dim=-1)[0] + 1e-10)
        elif self.max_measure[:5] == 'pmean':
            p = int(self.max_measure[-1])
            return - torch.log(1 - ((fs ** p).mean(dim=-1))**(1/p) + 1e-10)
        else:
            raise ValueError

    def forward_abox_ec(self, x, anon_e_emb):
        e_emb = self.e_embedding(x[:, :, 0])        
        r_emb = self.r_embedding(x[0, 0, 1])
        c_emb = self.c_embedding(x[:, 0, 2])
        r_fs = self._get_c_fs_batch((e_emb + r_emb).view(-1, e_emb.size()[-1]), anon_e_emb).view(e_emb.size()[0], e_emb.size()[1], -1)
        c_fs = self._get_c_fs_batch(c_emb, anon_e_emb).unsqueeze(dim=1)
        dofm = self._logical_exist(r_fs, c_fs)
        return ( - torch.log(dofm[:, 0] + 1e-10).mean() - torch.log(1 - dofm[:, 1:] + 1e-10).mean()) / 2

    def forward_abox_ec_created(self, x):
        e_emb = self.e_embedding(x[:, :, 0])
        c_emb = self.c_embedding(x[:, :, 1])
        emb = torch.cat([c_emb, e_emb], dim=-1)
        if self.cfg.loss_type == 'c':
            dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
            # dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
            return ( - torch.log(dofm[:, 0] + 1e-10).mean() - torch.log(1 - dofm[:, 1:] + 1e-10).mean()) / 2
        elif self.cfg.loss_type == 'r':
            dofm = self.fc_0(emb).squeeze(dim=-1)
            # dofm = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1)).squeeze(dim=-1)
            return - torch.nn.functional.logsigmoid(dofm[:, 0].unsqueeze(dim=-1) - dofm[:, 1:]).mean()
        else:
            raise ValueError

    def forward_ggi(self, x, stage='train'):
        e_1_emb = self.e_embedding(x[:, :, 0])
        r_emb = self.r_embedding(x[:, :, 1])
        e_2_emb = self.e_embedding(x[:, :, 2])
        emb = torch.cat([e_1_emb + r_emb, e_2_emb], dim=-1)
        if stage == 'train':
            if self.cfg.loss_type == 'c':
                dofm = torch.sigmoid(self.fc_0(emb)).squeeze(dim=-1)
                # dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
                return ( - torch.log(dofm[:, 0] + 1e-10).mean() - torch.log(1 - dofm[:, 1:] + 1e-10).mean()) / 2
            elif self.cfg.loss_type == 'r':
                dofm = self.fc_0(emb).squeeze(dim=-1)
                # dofm = self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1)).squeeze(dim=-1)
                return - torch.nn.functional.logsigmoid(dofm[:, 0].unsqueeze(dim=-1) - dofm[:, 1:]).mean()
            else:
                raise ValueError
        elif stage == 'test':
            return torch.sigmoid(self.fc_0(emb)).flatten()
            # return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).flatten()

def get_ranks(logits, y, already_ts_dict, already_hs_dict, flag):
    logits_sorted = torch.argsort(logits.cpu(), dim=-1, descending=True)
    if flag == 'head':
        ranks = ((logits_sorted == y[0]).nonzero()[0][0] + 1).long()
        key = (y[2].item(), y[1].item())
        try:
            already = already_hs_dict[key]
        except:
            already = None
    elif flag == 'tail':
        ranks = ((logits_sorted == y[2]).nonzero()[0][0] + 1).long()
        key = (y[0].item(), y[1].item())
        try:
            already = already_ts_dict[key]
        except:
            already = None
    ranking_better = logits_sorted[:ranks - 1]
    if already != None:
        for e in already:
            if (ranking_better == e).sum() == 1:
                ranks -= 1
    r = ranks.item()
    rr = (1 / ranks).item()
    h1 = (ranks == 1).float().item()
    h3 = (ranks <= 3).float().item()
    h10 = (ranks <= 10).float().item()
    return r, rr, h1, h3, h10

def ggi_evaluate(model, loader, e_dict, device, already_ts_dict, already_hs_dict):
    model.eval()
    mr, mrr, mh1, mh3, mh10 = 0, 0, 0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            if model.__class__.__name__ == 'FuzzyDL':
                logits = model.forward_ggi(X, stage='test')
            elif model.__class__.__name__ == 'KGCModel':
                logits = model.forward(X).flatten()
            logits_head, logits_tail = logits[:len(e_dict)], logits[len(e_dict):]
            r, rr, h1, h3, h10 = get_ranks(logits_head, y[0], already_ts_dict, already_hs_dict, flag='head')
            mr += r
            mrr += rr
            mh1 += h1
            mh3 += h3
            mh10 += h10
            r, rr, h1, h3, h10 = get_ranks(logits_tail, y[0], already_ts_dict, already_hs_dict, flag='tail')
            mr += r
            mrr += rr
            mh1 += h1
            mh3 += h3
            mh10 += h10
    counter = len(e_dict) * 2
    _, mrr, _, mh3, mh10 = round(mr/counter, 3), round(mrr/counter, 5), round(mh1/counter, 3), round(mh3/counter, 3), round(mh10/counter, 3)
    return mrr, mh3, mh10

def iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def compute_metrics(preds):
    n_pos = n_neg = len(preds) // 2
    labels = [0] * n_pos + [1] * n_neg
    
    mae_pos = round(sum(preds[:n_pos]) / n_pos, 4)
    auc = round(roc_auc_score(labels, preds), 4)
    aupr = round(average_precision_score(labels, preds), 4)
    
    precision, recall, _ = precision_recall_curve(labels, preds)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    fmax = round(np.max(f1_scores), 4)
    
    return mae_pos, auc, aupr, fmax

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Tuable
    parser.add_argument('--model', default='FALCON', type=str, help='FALCON, DistMult, TransE, ConvE')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--num_ng', default=8, type=int)
    parser.add_argument('--n_models', default=2, type=int)
    parser.add_argument('--loss_type', default='r', type=str, help='r for ranking, c for classification')
    parser.add_argument('--bs_kgc', default=64, type=int)
    parser.add_argument('--bs_ee', default=64, type=int)
    parser.add_argument('--bs_ec', default=64, type=int)
    parser.add_argument('--bs_tbox_name', default=64, type=int)
    parser.add_argument('--bs_tbox_desc', default=4, type=int)
    parser.add_argument('--anon_e', default=4, type=int)
    parser.add_argument('--n_e', default=1500, type=int)
    parser.add_argument('--n_abox_ec_created', default=1000, type=int)
    parser.add_argument('--n_inconsistent', default=0, type=int)
    parser.add_argument('--t_norm', default='product', type=str, help='product, minmax, Łukasiewicz')
    parser.add_argument('--residuum', default='notCorD', type=str)
    parser.add_argument('--max_measure', default='max', type=str)
    parser.add_argument('--scoring_fct_norm', default=2, type=float)  
    parser.add_argument('--kernel_size', default=3, type=int)           
    parser.add_argument('--convkb_drop_prob', default=0.2, type=float)  
    parser.add_argument('--out_channels', default=8, type=int)        
    # Untunable
    parser.add_argument('--data_root', default='../../data/pheno/', type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--valid_interval', default=1000, type=int)
    parser.add_argument('--tolerance', default=10, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    tbox_name_train, tbox_desc_train, \
        abox_ec, abox_ec_created, abox_ee_train, abox_ee_test, \
        c_dict, e_dict, e_dict_more, r_dict, all_alleles, all_omims, \
        already_ts_dict, already_hs_dict = get_data(cfg)
    print(f'Concept: {len(c_dict)}\tIndividual: {len(e_dict)}+{cfg.anon_e}+{len(abox_ec_created)}\tRelation: {len(r_dict)}', flush=True)
    print(f'TBox Name: {len(tbox_name_train)}\tTBox Description: {len(tbox_desc_train)}\t\
        ABox_ec: {len(abox_ec)}\tABox_ee_train:{len(abox_ee_train)}\tABox_ee_test:{len(abox_ee_test)}', flush=True)
    
    ee_dataset_train = EEDataset(cfg, abox_ee_train, e_dict, all_alleles, all_omims, already_ts_dict, already_hs_dict, stage='train')
    # ggi_dataset_test = EEDataset(cfg, abox_ee_test, e_dict, already_ts_dict, already_hs_dict, stage='test')
    # abox_ec_dataset = AboxECDataset(cfg, abox_ec, e_dict)
    # abox_ec_created_dataset = AboxECCreatedDataset(cfg, abox_ec_created, e_dict)
    # tbox_name_dataset = NaiveDataset(torch.tensor(tbox_name_train.values))
    # tbox_desc_dataset = NaiveDataset(tbox_desc_train)
    
    ee_dataloader_train = torch.utils.data.DataLoader(dataset=ee_dataset_train, 
                                                        batch_size=cfg.bs_ee,
                                                        num_workers=4,
                                                        shuffle=True,
                                                        drop_last=True)
    # ggi_dataloader_test = torch.utils.data.DataLoader(dataset=ggi_dataset_test, 
    #                                                     batch_size=1,
    #                                                     # num_workers=4,
    #                                                     shuffle=False,
    #                                                     drop_last=False)
    # abox_ec_dataloader = torch.utils.data.DataLoader(dataset=abox_ec_dataset, 
    #                                                     batch_size=cfg.bs_ec,
    #                                                     # num_workers=4,
    #                                                     shuffle=True,
    #                                                     drop_last=True)
    # abox_ec_created_dataloader = torch.utils.data.DataLoader(dataset=abox_ec_created_dataset, 
    #                                                     batch_size=cfg.bs_ec,
    #                                                     # num_workers=4,
    #                                                     shuffle=True,
    #                                                     drop_last=True)
    # tbox_name_dataloader = torch.utils.data.DataLoader(dataset=tbox_name_dataset, 
    #                                                     batch_size=cfg.bs_tbox_name,
    #                                                     shuffle=True,
    #                                                     drop_last=True)
    # tbox_desc_dataloader = torch.utils.data.DataLoader(dataset=tbox_desc_dataset, 
    #                                                     batch_size=cfg.bs_tbox_desc,
    #                                                     shuffle=True,
    #                                                     drop_last=True)

    # ggi_dataloader_train = iterator(ggi_dataloader_train)
    # abox_ec_dataloader = iterator(abox_ec_dataloader)
    # abox_ec_created_dataloader = iterator(abox_ec_created_dataloader)
    # tbox_name_dataloader = iterator(tbox_name_dataloader)
    # tbox_desc_dataloader = iterator(tbox_desc_dataloader)
    
    # device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    # model = FALCON(c_dict, e_dict_more, r_dict, cfg, device)
    # model = model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    
    # weights = [5, 1, 1, 1, 1]
    # if cfg.verbose:
    #     ranger = tqdm.tqdm(range(cfg.max_steps))
    # else:
    #     ranger = range(cfg.max_steps)
    # losses = []
    # results = []
    # for step in ranger:
    #     model.train()
    #     # anonymous entities
    #     anon_e_emb_1 = model.e_embedding.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim)).to(device)
    #     anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim).to(device)
    #     torch.nn.init.xavier_uniform_(anon_e_emb_2)
    #     anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)

    #     # ggi loss
    #     loss_ggi = model.forward_ggi(next(ggi_dataloader_train).to(device).long())
        
    #     # abox_ec loss
    #     loss_abox_ec = model.forward_abox_ec(next(abox_ec_dataloader).to(device).long(), anon_e_emb)

    #     # abox_ec_created loss
    #     loss_abox_ec_created = model.forward_abox_ec_created(next(abox_ec_created_dataloader).to(device).long())
        
    #     # tbox_name loss
    #     tbox_name_batch = next(tbox_name_dataloader).to(device).long()
    #     loss_tbox_name = model.forward_name(tbox_name_batch, anon_e_emb)
        
    #     # tbox_description loss
    #     loss_tbox_desc = []
    #     for axiom in next(tbox_desc_dataloader):
    #         fs = model.forward(axiom, anon_e_emb)
    #         loss_tbox_desc.append(model.get_cc_loss(fs))
    #     loss_tbox_desc = sum(loss_tbox_desc) / len(loss_tbox_desc)
        
    #     loss = (loss_ggi * weights[0] + loss_abox_ec * weights[1] + loss_abox_ec_created * weights[2] + loss_tbox_name * weights[3] + loss_tbox_desc * weights[4]) / sum(weights)
    #     # if torch.isnan(loss):
    #     #     pdb.set_trace()
    #     # if (step + 1) % (cfg.valid_interval // 10) == 0:
    #     #     print(round(loss_ggi.item(), 4), round(loss_abox_ec.item(), 4), round(loss_abox_ec_created.item(), 4), round(loss_tbox_name.item(), 4), round(loss_tbox_desc.item(), 4), round(loss.item(), 4))
        
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     losses.append(loss.item())
        
    #     if (step + 1) % cfg.valid_interval == 0:
    #         avg_loss = round(sum(losses)/len(losses), 4)
    #         print(f'Loss: {avg_loss}', flush=True)
    #         losses = []
    #         mrr, mh3, mh10 = ggi_evaluate(model, ggi_dataloader_test, e_dict, device, already_ts_dict, already_hs_dict)
    #         print(f'#GGI# MRR: {round(mrr, 3)}, H3: {mh3}, H10: {mh10}', flush=True)
    #         model.eval()
    #         with torch.no_grad():
    #             preds = []
    #             for axiom in tbox_test_pos:
    #                 fs = model.forward(axiom, anon_e_emb)
    #                 preds.append(max(fs).item())
    #             for axiom in tbox_test_neg:
    #                 fs = model.forward(axiom, anon_e_emb)
    #                 preds.append(max(fs).item())
    #         mae_pos, auc, aupr, fmax = compute_metrics(preds)
    #         print(f'#TBox# MAE:{mae_pos}\tAUC:{auc}\tAUPR:{aupr}\tFmax:{fmax}')
    #         results.append([mrr, mh3, mh10, mae_pos, auc, aupr, fmax])
    # results = torch.tensor(results)
    # final_results = results[results.transpose(1, 0).max(dim=-1)[1][0]]
    # print(results)
    # mrr, mh3, mh10, mae_pos, auc, aupr, fmax = final_results
    # print(f'Best: #GGI# MRR: {round(mrr.item(), 3)}\tH3: {round(mh3.item(), 3)}\tH10: {round(mh10.item(), 3)}', flush=True)
    # print(f'Best: #TBox# MAE:{round(mae_pos.item(), 3)}\tAUC:{round(auc.item(), 3)}\tAUPR:{round(aupr.item(), 3)}\tFmax:{round(fmax.item(), 3)}')

