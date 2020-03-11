import argparse
import pickle
from pathlib import Path
from datetime import date
from datetime import datetime
from socket import gethostname
import os
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, recall_score, precision_score

from phenotools import factor_matrices_to_xlsx, extract_corrs_to_xlsx, sparsity_similarity

from hitf import CollectiveHITF


def mortality_prediction(pt_rep_train, pt_rep_test, mortality_train, mortality_test):
    lr = LogisticRegressionCV(cv=5, class_weight='balanced', solver='liblinear')
    lr.fit(pt_rep_train, mortality_train)

    pred_prob = lr.predict_proba(pt_rep_test)[:, 1]
    mortality_pred = lr.predict(pt_rep_test)

    precision_, recall_, thr_ = precision_recall_curve(mortality_test, pred_prob)

    mortality_pred_scores = {
        'auc': roc_auc_score(mortality_test, pred_prob),
        'precision': precision_score(mortality_test, mortality_pred),
        'recall': recall_score(mortality_test, mortality_pred),
        'f1': f1_score(mortality_test, mortality_pred),
        'prauc': auc(recall_, precision_)
    }
    return mortality_pred_scores



def evaluation(indata, factors, idx_train, pt_rep_train, pt_rep_test,
               mortality_train, mortality_test,
               out_path):

    mortality_pred_scores = mortality_prediction(pt_rep_train, pt_rep_test,
                                                 mortality_train, mortality_test)
    print('\nmortality prediction:')
    print('\n'.join(['  '+key+': '+str(value) for key, value in mortality_pred_scores.items()]), end='\n\n')

    ########################
    # interpret phenotypes #
    ########################
    item_dicts = {
        'dx': ('Diagnoses', indata['dx_idx2desc']),
        'rx': ('Medications', indata['rx_idx2desc']),
        'lab': ('Lab tests', indata['lab_idx2desc'])
    }
    if 'input_idx2desc' in indata.keys():
        item_dicts['input'] = ('Input Fluids', indata['input_idx2desc'])

    modalities = list(factors.keys())

    item_dicts = [item_dicts[k] for k in modalities]  # exclude vital signs in evaluation for now.
    phenotype_factors = [factors[k] for k in modalities]
    filepath = out_path / f'phenotypes-{gethostname()}-{os.getpid()}.xlsx'
    factor_matrices_to_xlsx(phenotype_factors, item_dicts, filepath, threshold=[1e-4] * 4)

    ###########################################
    # interpret Dx/Rx & Dx/Lab correspondence #
    ###########################################
    if 'dx' in modalities and 'rx' in modalities:
        filepath = out_path / f'correspondence-dxrx-{gethostname()}-{os.getpid()}.xlsx'
        extract_corrs_to_xlsx(indata['D'][idx_train, :],
                              pt_rep_train, factors['dx'], factors['rx'],
                              indata['dx_idx2desc'],
                              indata['rx_idx2desc'],
                              filepath, ws_name=None)

    if 'dx' in modalities and 'lab' in modalities:
        filepath = out_path / f'correspondence-dxlab-{gethostname()}-{os.getpid()}.xlsx'
        extract_corrs_to_xlsx(indata['D'][idx_train, :],
                              pt_rep_train, factors['dx'], factors['lab'],
                              indata['dx_idx2desc'],
                              indata['lab_idx2desc'],
                              filepath, ws_name=None)

    if 'rx' in modalities and 'lab' in modalities:
        filepath = out_path / f'correspondence-rxlab-{gethostname()}-{os.getpid()}.xlsx'
        M_binarized = indata['M'][idx_train, :]
        M_binarized[M_binarized > 0] = 0
        extract_corrs_to_xlsx(M_binarized,
                              pt_rep_train, factors['rx'], factors['lab'],
                              indata['rx_idx2desc'],
                              indata['lab_idx2desc'],
                              filepath, ws_name=None)
    
    if 'dx' in modalities and 'input' in modalities:
        filepath = out_path / f'correspondence-dxinput-{gethostname()}-{os.getpid()}.xlsx'
        extract_corrs_to_xlsx(indata['D'][idx_train, :],
                              pt_rep_train, factors['dx'], factors['input'],
                              indata['dx_idx2desc'],
                              indata['input_idx2desc'],
                              filepath, ws_name=None)

    # similarity and sparsity.
    sparsity, similarity = sparsity_similarity(phenotype_factors)
    print('sparsity:', sparsity, sum(sparsity)/len(sparsity))
    print('similarity:', similarity, sum(similarity)/len(similarity))

    # save results to file
    evaluation = {
        'mortality_prediction': {k: float(v) for k,  v in mortality_pred_scores.items()},
        'sparsity': [float(v) for v in sparsity],
        'similarity': [float(v) for v in similarity],
        'avg_sparsity': sum(sparsity[1:])/len(sparsity[1:]),
        'avg_similarity': sum(similarity[1:])/len(similarity[1:])
    }

    with open(out_path / 'evaluation.json', 'w') as f:
        json.dump(evaluation, f)


def chitf_train_evaluate(exp_id,
                         data_path,
                         out_path,
                         rank,
                         modalities=['dx-rx', 'dx-lab', 'dx-input'],
                         distributions=['P', 'P', 'G'],
                         weights=[1, 1, 1],
                         init_lr=0.0001,
                         weight_decay=0,
                         angular_weight=1,
                         angular_threshold=1,
                         elastic_weight=1,
                         elastic_l1_ratio=0.5,
                         gaussian_var=1,
                         train_size=0.8,
                         test_size=0.2,
                         max_iters=1000000,
                         max_proj_iters=6000,
                         device=None,
                         seed=None):

    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load data
    with open(data_path, 'rb') as f:
        indata = pickle.load(f)

    idx_train, idx_test, mortality_train, mortality_test = train_test_split(np.arange(len(indata['mortality'])),
                                                                            indata['mortality'],
                                                                            train_size=train_size,
                                                                            test_size=test_size,
                                                                            random_state=seed)
    inputs = {
        'dx': (torch.FloatTensor(indata['D']).to(device), True),
        'rx': (torch.FloatTensor(indata['M']).to(device), False),
        'lab': (torch.FloatTensor(indata['L']).to(device), False),
    }

    if 'T' in indata.keys():
        input_fluids = torch.FloatTensor(indata['T']).to(device)
        inputs['input'] = (input_fluids, False)

    modes_involved = []
    for modes in modalities:
        modes_involved += modes.split('-')
    modes_involved = list(set(modes_involved))

    inputs_train = {k: (X[torch.LongTensor(idx_train)], bin_)
                    for k, (X, bin_) in inputs.items()
                    if k in modes_involved}
    inputs_test = {k: (X[torch.LongTensor(idx_test)], bin_)
                   for k, (X, bin_) in inputs.items()
                   if k in modes_involved}

    print(modalities)
    print(f'Data loaded from: {data_path}')
    print(f'  Training size ratio: {train_size:.0%}')
    print(f'  Test size ratio: {test_size:.0%}')
    print('  Sizes of training data: ', [X.shape for k, (X, t) in inputs_train.items()])
    print()
    print('Configurations:')
    print(f'  Hidden Tensors: {"; ".join([f"{m}({d})" for m,d in zip(modalities, distributions)])}')
    print(f'  Rank: {rank}')
    print(f'  Learning Rate: {init_lr}')
    print(f'  Weight Decay: {weight_decay}')
    print(f'  Angular Regularization: weight={angular_weight}; threshold={angular_threshold}')
    print(f'  Elastic Net Regularization: weight={elastic_weight}; l1 ratio={elastic_l1_ratio}')
    if 'G' in distributions:
        print(f'  Variance of Gaussian Distribution: {gaussian_var}')
    print(f'\nStart fitting cHITF model at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # create model
    exp_id += f'_{",".join(modalities)}_R{rank}'

    out_path = Path(out_path) / f'{date.today():%Y%m%d}-{exp_id}-{gethostname()}-{os.getpid()}'
    out_path.mkdir(parents=True)

    writer = SummaryWriter(out_path)

    gaussian_loss_kwargs = {'var': gaussian_var}
    chitf = CollectiveHITF(inputs=inputs_train, 
                           modalities=modalities, 
                           distributions=distributions,
                           rank=rank, 
                           device=device,
                           labels=mortality_train,
                           weights=weights,
                           gaussian_loss_kwargs=gaussian_loss_kwargs,
                           angular_weight=angular_weight,
                           angular_threshold=angular_threshold,
                           elastic_weight=elastic_weight,
                           elastic_l1_ratio=elastic_l1_ratio,
                           init_factors=None,
                           tb_writer=writer)
    chitf.fit(lr_init=init_lr, weight_decay=weight_decay, max_iters=max_iters)

    print('\n\nProjection:')
    projector = chitf.construct_projector(inputs_test)
    projector.fit(max_iters=max_proj_iters, lr_init=0.1)

    pt_rep_train = chitf.factors.pt_reps.data.cpu().numpy()
    pt_rep_test = projector.factors.pt_reps.data.cpu().numpy()

    chitf.save_factors(out_path / 'factors.pt')

    factors = {k: v.data.cpu().numpy() for k, v in chitf.factors.factors.items()}

    evaluation(indata, factors, idx_train, pt_rep_train, pt_rep_test,
               mortality_train, mortality_test,
               out_path)

    np.savez(out_path / 'patient_reps.npz',
             pt_rep_train=pt_rep_train, pt_rep_test=pt_rep_test,
             mortality_train=mortality_train, mortality_test=mortality_test)

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cHITF Framework for Computational Phenotyping')

    parser.add_argument('exp_id', type=str)
    parser.add_argument('-d', '--distributions', type=str, nargs='+', default=['P', 'P', 'G'])
    parser.add_argument('-M', '--modalities', type=str, nargs='+', default=['dx-rx', 'dx-lab', 'dx-input'])
    parser.add_argument('-w', '--weights', type=float, nargs='+', default=[1, 1, 1])
    parser.add_argument('-i', '--input', type=str, default='./data/carevue-dx,rx,lab,input-20191003.pkl',
                        help='Path of input data.')
    parser.add_argument('-o', '--output', type=str, default='./results/')
    parser.add_argument('-R', '--rank', type=int, default=50,
                        help='Specify the number of phenotypes.')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--angular_threshold', type=float, default=0.5)
    parser.add_argument('--angular_weight', type=float, default=1)
    parser.add_argument('--elastic_weight', type=float, default=1e-3)
    parser.add_argument('--l1_ratio', type=float, default=0.7)
    parser.add_argument('-v', '--gaussian_var', type=float, default=1e-9)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--max_proj_iters', type=int, default=3000)

    args = parser.parse_args()

    chitf_train_evaluate(args.exp_id,
                         data_path=args.input,
                         out_path=args.output,
                         rank=args.rank,
                         modalities=args.modalities,
                         distributions=args.distributions,
                         weights=args.weights,
                         init_lr=args.lr,
                         weight_decay=args.weight_decay,
                         seed=args.seed,
                         angular_threshold=args.angular_threshold,
                         angular_weight=args.angular_weight,
                         elastic_weight=args.elastic_weight,
                         elastic_l1_ratio=args.l1_ratio,
                         gaussian_var=args.gaussian_var,
                         max_iters=args.max_iters,
                         max_proj_iters=args.max_proj_iters)