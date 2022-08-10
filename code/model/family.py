import torch
import pdb
import argparse
import re
import pandas as pd
import numpy as np
import random

def get_all_concepts_and_relations(axiom, all_concepts, all_relations):
    if axiom[0] == '<':
        if axiom != '<Nothing>':
            all_concepts.add(axiom)

    elif axiom[:15] == 'ObjectDisjoint(':
        axiom_parts = axiom.split(' ')
        counter = 0
        for i, part in enumerate(axiom_parts):
            counter += part.count('(')
            counter -= part.count(')')
            if counter == 1:
                ridge = i
        left = ' '.join(axiom_parts[:ridge + 1])[15:]
        right = ' '.join(axiom_parts[ridge + 1:])[:-1]
        get_all_concepts_and_relations(left, all_concepts, all_relations)
        get_all_concepts_and_relations(right, all_concepts, all_relations)
    
    elif axiom[:21] == 'ObjectIntersectionOf(':
        axiom_parts = axiom.split(' ')
        counter = 0
        for i, part in enumerate(axiom_parts):
            counter += part.count('(')
            counter -= part.count(')')
            if counter == 1:
                ridge = i
        left = ' '.join(axiom_parts[:ridge + 1])[21:]
        right = ' '.join(axiom_parts[ridge + 1:])[:-1]
        get_all_concepts_and_relations(left, all_concepts, all_relations)
        get_all_concepts_and_relations(right, all_concepts, all_relations)
        
    elif axiom[:14] == 'ObjectUnionOf(':
        axiom_parts = axiom.split(' ')
        counter = 0
        for i, part in enumerate(axiom_parts):
            counter += part.count('(')
            counter -= part.count(')')
            if counter == 1:
                ridge = i
        left = ' '.join(axiom_parts[:ridge + 1])[14:]
        right = ' '.join(axiom_parts[ridge + 1:])[:-1]
        get_all_concepts_and_relations(left, all_concepts, all_relations)
        get_all_concepts_and_relations(right, all_concepts, all_relations)
        
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
    

class FALCON(torch.nn.Module):
    def __init__(self, emb_dim, c_dict, e_dict, r_dict, cfg):
        super().__init__()
        self.c_dict = c_dict
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.n_entity = len(e_dict) + cfg.anon_e
        self.anon_e = cfg.anon_e
        self.c_embedding = torch.nn.Embedding(len(c_dict), emb_dim)
        self.r_embedding = torch.nn.Embedding(len(r_dict), emb_dim)
        self.e_embedding = torch.nn.Embedding(len(e_dict), emb_dim)
        self.fc_0 = torch.nn.Linear(emb_dim * 2, emb_dim)
        self.fc_1 = torch.nn.Linear(emb_dim, 1)
        torch.nn.init.xavier_uniform_(self.c_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_0.weight.data)
        torch.nn.init.xavier_uniform_(self.fc_1.weight.data)
        self.max_measure = cfg.max_measure
        self.t_norm = cfg.t_norm
        self.nothing = torch.zeros(self.n_entity)
    
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
    
    def _logical_exist(self, r_fs, c_fs):
        return self._logical_and(r_fs, c_fs).max(dim=-1)[0]

    def _get_c_fs(self, c_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0)
        c_emb = c_emb.expand_as(e_emb)
        emb = torch.cat([c_emb, e_emb], dim=-1)
        return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        
    def _get_r_fs(self, r_emb, anon_e_emb):
        e_emb = torch.cat([self.e_embedding.weight, anon_e_emb], dim=0)
        l_emb = (e_emb + r_emb.unsqueeze(dim=0)).unsqueeze(dim=1).repeat(1, self.n_entity, 1)
        r_emb = e_emb.unsqueeze(dim=0).repeat(self.n_entity, 1, 1)
        emb = torch.cat([l_emb, r_emb], dim=-1)
        return torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)

    def forward(self, axiom, anon_e_emb):

        if axiom[0] == '<':
            if axiom == '<Nothing>':
                return self.nothing
            else:
                c_emb = self.c_embedding(torch.tensor(self.c_dict[axiom]))
                return self._get_c_fs(c_emb, anon_e_emb)

        elif axiom[:21] == 'ObjectIntersectionOf(':
            axiom_parts = axiom.split(' ')
            counter = 0
            for i, part in enumerate(axiom_parts):
                counter += part.count('(')
                counter -= part.count(')')
                if counter == 1:
                    ridge = i
            left = ' '.join(axiom_parts[:ridge + 1])[21:]
            right = ' '.join(axiom_parts[ridge + 1:])[:-1]
            return self._logical_and(self.forward(left, anon_e_emb), self.forward(right, anon_e_emb))

        elif axiom[:14] == 'ObjectUnionOf(':
            axiom_parts = axiom.split(' ')
            counter = 0
            for i, part in enumerate(axiom_parts):
                counter += part.count('(')
                counter -= part.count(')')
                if counter == 1:
                    ridge = i
            left = ' '.join(axiom_parts[:ridge + 1])[14:]
            right = ' '.join(axiom_parts[ridge + 1:])[:-1]
            return self._logical_or(self.forward(left, anon_e_emb), self.forward(right, anon_e_emb))

        elif axiom[:22] == 'ObjectSomeValuesFrom(<':
            axiom_parts = axiom.split(' ')
            r_emb = self.r_embedding(torch.tensor(self.r_dict[axiom_parts[0][21:]]))
            r_fs = self._get_r_fs(r_emb, anon_e_emb)
            right = ' '.join(axiom_parts[1:])[:-1]
            return self._logical_exist(r_fs, self.forward(right, anon_e_emb))

        elif axiom[:19] == 'ObjectComplementOf(':
            return self._logical_not(self.forward(axiom[19:-1], anon_e_emb))

        else:
            raise ValueError

    def get_cc_loss(self, fs):
        if self.max_measure == 'max':
            return - torch.log(1 - fs.max(dim=-1)[0] + 1e-10)
            # return (fs.max(dim=-1)[0] - 0) ** 2
        elif self.max_measure[:5] == 'pmean':
            p = int(self.max_measure[-1])
            return ((fs ** p).mean(dim=-1))**(1/p)
        else:
            raise ValueError

    def get_ec_loss(self, axiom):
        axiom = axiom.split(' ')
        e, c = axiom[0], axiom[1]
        e_id = self.e_dict[e]
        c_id = self.c_dict[c]
        e_emb = self.e_embedding.weight[e_id]
        c_emb = self.c_embedding.weight[c_id]
        emb = torch.cat([c_emb, e_emb])
        dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        return - torch.log(dofm + 1e-10) 

    def get_ee_loss(self, axiom):
        axiom = axiom.split(' ')
        e_1, r, e_2 = axiom[0], axiom[1], axiom[2]
        e_1_id = self.e_dict[e_1]
        r_id = self.r_dict[r]
        e_2_id = self.e_dict[e_2]
        e_1_emb = self.e_embedding.weight[e_1_id]
        r_emb = self.r_embedding.weight[r_id]
        e_2_emb = self.e_embedding.weight[e_2_id]
        emb = torch.cat([e_1_emb + r_emb, e_2_emb])
        dofm = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb), negative_slope=0.1))).squeeze(dim=-1)
        
        emb_neg = torch.cat([e_2_emb + r_emb, e_1_emb])
        dofm_neg = torch.sigmoid(self.fc_1(torch.nn.functional.leaky_relu(self.fc_0(emb_neg), negative_slope=0.1))).squeeze(dim=-1)
        return - torch.log(dofm + 1e-10) - torch.log(1 - dofm_neg + 1e-10) 

    def predict(self, anon_e_emb):
        cols = ['concept']
        lines = []
        for _, concept in enumerate(c_dict):
            concept_id = self.c_dict[concept]
            c_emb = self.c_embedding.weight.data[concept_id]
            c_fs = self._get_c_fs(c_emb, anon_e_emb)
            line = [concept]
            line.extend([round(i, 1) for i in c_fs.detach().numpy().tolist()])
            lines.append(line)
        cols.extend(self.e_dict.keys())
        for i in range(self.anon_e):
            cols.append(f'anon_{i}')
        return pd.DataFrame(lines, columns=cols)
            

def get_data():
    tbox = [
        'ObjectIntersectionOf(<Male> ObjectComplementOf(<Person>))',
        'ObjectIntersectionOf(<Female> ObjectComplementOf(<Person>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Male> <Female>) ObjectComplementOf(<Nothing>))',
        
        'ObjectIntersectionOf(<Parent> ObjectComplementOf(<Person>))',
        'ObjectIntersectionOf(<Child> ObjectComplementOf(<Person>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Parent> <Child>) ObjectComplementOf(<Nothing>))',
        
        'ObjectIntersectionOf(<Father> ObjectComplementOf(<Male>))',
        'ObjectIntersectionOf(<Boy> ObjectComplementOf(<Male>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Father> <Boy>) ObjectComplementOf(<Nothing>))',
        
        'ObjectIntersectionOf(<Mother> ObjectComplementOf(<Female>))',
        'ObjectIntersectionOf(<Girl> ObjectComplementOf(<Female>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Mother> <Girl>) ObjectComplementOf(<Nothing>))',
        
        'ObjectIntersectionOf(<Father> ObjectComplementOf(<Parent>))',
        'ObjectIntersectionOf(<Mother> ObjectComplementOf(<Parent>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Father> <Mother>) ObjectComplementOf(<Nothing>))',
        
        'ObjectIntersectionOf(<Boy> ObjectComplementOf(<Child>))',
        'ObjectIntersectionOf(<Girl> ObjectComplementOf(<Child>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Boy> <Girl>) ObjectComplementOf(<Nothing>))',
        
        'ObjectIntersectionOf(ObjectIntersectionOf(<Female> <Parent>) ObjectComplementOf(<Mother>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Male> <Parent>) ObjectComplementOf(<Father>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Female> <Child>) ObjectComplementOf(<Girl>))',
        'ObjectIntersectionOf(ObjectIntersectionOf(<Male> <Child>) ObjectComplementOf(<Boy>))',
        
        'ObjectIntersectionOf(ObjectSomeValuesFrom(<hasChild> <Person>) ObjectComplementOf(<Parent>))',
        'ObjectIntersectionOf(ObjectSomeValuesFrom(<hasParent> <Person>) ObjectComplementOf(<Child>))',
        
        'ObjectIntersectionOf(<Grandma> ObjectComplementOf(<Mother>))',
        
    ]
    abox_ec = [
        # '<person_1> <Mother>',
        # '<male_1> <Boy>',
        # '<female_1> <Girl>',
    ]
    abox_ee = [
        # '<father_1> <hasChild> <boy_1>',
        # '<father_1> <hasChild> <girl_1>',
        # '<mother_1> <hasChild> <boy_1>',
        # '<mother_1> <hasChild> <girl_1>',
        # '<father_1> <hasChild> <male_1>',
        # '<mother_1> <hasChild> <male_1>',
        
        # '<boy_1> <hasParent> <mother_1>',
        # '<girl_1> <hasParent> <mother_1>',
        # '<boy_1> <hasParent> <father_1>',
        # '<girl_1> <hasParent> <father_1>',
        # '<male_1> <hasParent> <mother_1>',
        # '<male_1> <hasParent> <father_1>',
        
    ]
    all_concepts = set()
    all_relations = set()
    for axiom in tbox:
        get_all_concepts_and_relations(axiom, all_concepts, all_relations)
    c_dict = {k: v for v, k in enumerate(all_concepts)}
    r_dict = {k: v for v, k in enumerate(all_relations)}
    for concept in all_concepts:
        abox_ec.append(concept.lower()[:-1] + '_0' + concept[-1:] + ' ' + concept)
            
    all_entities = set()
    for axiom in abox_ec:
        all_entities.add(axiom.split(' ')[0])
    for axiom in abox_ee:
        all_entities.add(axiom.split(' ')[0])
        all_entities.add(axiom.split(' ')[-1])
    e_dict = {k: v for v, k in enumerate(all_entities)}
    
    return tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict


def get_one_model(cfg, tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict):
    model = FALCON(emb_dim=cfg.emb_dim, c_dict=c_dict, e_dict=e_dict, r_dict=r_dict, cfg=cfg)
    params = list(model.c_embedding.parameters()) + list(model.e_embedding.parameters()) + list(model.r_embedding.parameters()) + list(model.fc_0.parameters()) + list(model.fc_1.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.wd)
    for epoch in range(cfg.max_epochs):
        anon_e_emb_1 = model.e_embedding.weight.detach()[:cfg.anon_e//2] + torch.normal(0, 0.1, size=(cfg.anon_e//2, cfg.emb_dim))
        anon_e_emb_2 = torch.rand(cfg.anon_e//2, cfg.emb_dim)
        torch.nn.init.xavier_uniform_(anon_e_emb_2)
        anon_e_emb = torch.cat([anon_e_emb_1, anon_e_emb_2], dim=0)
        model.train()
        loss_cc = []
        loss_ec = []
        loss_ee = []
        for axiom in tbox:
            fs = model.forward(axiom, anon_e_emb)
            loss_cc.append(model.get_cc_loss(fs))
            
        for axiom in abox_ec:
            loss_ec.append(model.get_ec_loss(axiom))
        for axiom in abox_ee:
            loss_ee.append(model.get_ee_loss(axiom))
        loss = 0.5 * (sum(loss_cc) / len(loss_cc)) + 0.5 * (sum(loss_ec) / len(loss_ec)) 
        # + 0.2 * (sum(loss_ee) / len(loss_ee)) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}:')
            print(f'Loss: {round(loss.item(), 4)}')
        # print(model.forward('ObjectIntersectionOf(<Parent> <Child>)'))
        # print(model.e_embedding.weight.data[0][0])
    with torch.no_grad():
        model.eval()
        ret = model.predict(anon_e_emb)
        try:
            ret = ret.drop(ret[ret['concept']=='<Everything>'].index)
            ret.drop('everything', inplace=True, axis=1)
        except:
            pass
        # mid = ret[['concept', '<mother_0>']]
        # if mid[mid['concept'] == '<Grandma>']['<mother_0>'].sum() == 0:
        #     print(ret)
        #     pdb.set_trace()
        print(ret)
        # cols = [' ']
        # cols.extend(e_dict.keys())
        # rows = np.expand_dims(np.array(list(e_dict.keys())), axis=-1)
        # for rel in r_dict.keys():
        #     rel_id = r_dict[rel]
        #     r_emb = model.e_embedding.weight.data[rel_id]
        #     r_fs = model._get_r_fs(r_emb)
        #     print('***************')
        #     print(rel)
        #     print('***************')
        #     ret_rel = pd.DataFrame(np.c_[rows, r_fs.numpy()], columns=cols)
        #     print(ret_rel)
    return ret

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--emb_dim', default=50, type=int)
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--max_measure', default='max')
    parser.add_argument('--t_norm', default='product', type=str, help='product, minmax, Łukasiewicz, or NN')
    parser.add_argument('--n_models', default=2, type=int)
    parser.add_argument('--anon_e', default=4, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:')
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}')
    tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict = get_data()
    # rets = []
    for i in range(cfg.n_models):
        ret = get_one_model(cfg, tbox, abox_ec, abox_ee, c_dict, e_dict, r_dict)
        print(f'{i+1} Models Finished')
        # rets.append(torch.tensor(ret.values[:, 1:].tolist()).unsqueeze(dim=-2))
        # ret.to_csv(f'../log/family_{i+1}.csv')
    # cols = ret.columns
    # rows = np.expand_dims(ret.values[:, 0], axis=-1)
    # rets = torch.cat(rets, dim=-2).max(dim=-2)[0]
    # rets = pd.DataFrame(np.c_[rows, rets.numpy()], columns=cols)
    # print(rets)
    # ret.to_csv('../log/family_multi.csv')
    # pdb.set_trace()     
    