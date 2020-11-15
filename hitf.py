"""
Implementation of Hidden Interaction Tensor Factorization (HITF) model based on
pytorch.

"""
import time
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


class LatentFactors(nn.Module):
    def __init__(self, num_pt, modality_size_dict, rank):
        super().__init__()
        self.pt_reps = nn.Parameter(torch.rand(num_pt, rank))
        self.factors = nn.ParameterDict({modality: nn.Parameter(torch.rand(size, rank))
                                         for modality, size in modality_size_dict.items()})
        self._initialize_factors()
        self.non_negative_projection()

    def _initialize_factors(self):
        # self.pt_reps.data = self.pt_reps.data / self.pt_reps.data.sum(dim=0)
        # for _, X in self.factors.items():
        #     X.data = X.data / X.data.sum(dim=0)
        pass

    def non_negative_projection(self):
        self.pt_reps.data[self.pt_reps.data < 0] = 0
        for k, X in self.factors.items():
            X.data[X.data < 0] = 0


class HiddenInteractionTensor(nn.Module):
    def __init__(self, modalities, factors):
        super().__init__()
        self.modalities = modalities
        self.pt_reps = factors.pt_reps
        self.factors = [factors.factors[x] for x in modalities]
        self.num_mode = len(modalities)

    def reconstruct_nth_marginalization(self, n):
        assert 0 <= n < self.num_mode
        V_hat = self.pt_reps
        for i, U in enumerate(self.factors):
            if i != n:
                V_hat @= torch.diag(U.sum(dim=0))
        V_hat @= self.factors[n].t()
        return V_hat

    def forward(self):
        return [self.reconstruct_nth_marginalization(n) for n in range(self.num_mode)]


class PoissonNegativeLikelihood(nn.Module):
    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight
    def forward(self, reconstruction, target, bin_target: bool):
        if bin_target:
            log_number = ((1 - torch.exp(-reconstruction)) / torch.exp(-reconstruction).clamp(min=1e-15))
        else:
            log_number = reconstruction
        nll_loss = reconstruction - target * torch.log(log_number.clamp(min=1e-15))
        nll_loss = nll_loss + target * (torch.log(target.clamp(min=1e-15)) - 1)
        pt_loss = nll_loss.mean(dim=1)  # return loss per patient
        return pt_loss


class GaussianNegativeLikelihood(nn.Module):
    def __init__(self, dim_sum, var=1, gamma=0, weight=1):
        super().__init__()
        self.var = Parameter(torch.FloatTensor([var]), requires_grad=False)
        self.gamma = Parameter(torch.FloatTensor([gamma]), requires_grad=False)
        self.dim_sum = dim_sum
        self.weight = weight

    def forward(self, reconstruction, target, bin_target):
        if bin_target:
            ber_prob = 1 - (1 + torch.erf((self.gamma - reconstruction) / (2 * self.var) ** 0.5)) / 2
            nll_loss = - target * torch.log(ber_prob + 1e-15) - (1 - target) * torch.log(
                (1 - ber_prob) + 1e-15)
        else:
            nll_loss = ((target - reconstruction) ** 2) / (2 * self.dim_sum * self.var) + torch.log(
                (self.dim_sum * self.var) + 1e-15) / 2

        pt_loss = self.weight * nll_loss.mean(dim=1)  # return loss per patient
        return pt_loss


class ElasticNet(nn.Module):
    def __init__(self, l1_ratio=0.5):
        super().__init__()
        self.l1_ratio = l1_ratio

    def forward(self, X):
        reg = (1 - self.l1_ratio) * torch.norm(X, p='fro') ** 2
        reg += self.l1_ratio * X.abs().sum()
        return reg


class PairwiseAngularRegularization(nn.Module):
    def __init__(self, rank, threshold=0):
        super().__init__()
        self.rank = rank
        self.threshold = threshold
        self.eye_matrix = Parameter(torch.eye(rank), requires_grad=False)

    def forward(self, X):
        if self.threshold == 1:
            return torch.FloatTensor([0]).to(X.device)
        normalized_X = X / (X.norm(dim=0) + 1e-15)
        cos_sim = normalized_X.t() @ normalized_X - self.eye_matrix.to(X.device) - self.threshold
        cos_sim = torch.where(cos_sim>0, cos_sim, torch.zeros_like(cos_sim))
        return cos_sim.sum() / (self.rank * (self.rank-1))


class CollectiveHITF(object):
    def __init__(self,
                 inputs: dict,  # {modality_name: (obs, bin_flag)}
                 modalities,  # e.g. ['dx-rx', 'dx-lab', 'dx-vital']
                 distributions,  # e.g. ['P', 'P', 'G']
                 rank,
                 device,
                 labels=None,
                 weights=None,
                 gaussian_loss_kwargs={},
                 angular_weight=1,
                 angular_threshold=1,
                 elastic_weight=1,
                 elastic_l1_ratio=0.5,
                 init_factors=None,
                 tb_writer=None,
                 projector=False):

        self.inputs = inputs
        self.modalities = modalities
        self.distributions = distributions
        self.rank = rank
        self.device = device
        self.gaussian_loss_kwargs = gaussian_loss_kwargs

        self.num_pt = inputs[list(inputs.keys())[0]][0].shape[0]
        self.modality_size_dict = {name: X.shape[1] for name, (X, bin_) in inputs.items()}

        if init_factors is None:
            init_factors = LatentFactors(num_pt=self.num_pt,
                                         modality_size_dict=self.modality_size_dict,
                                         rank=rank).to(device)
        self.factors = init_factors
        self.projector = projector

        self.hidden_tensors = {}
        if weights is None:
            weights = [1] * len(modalities)
        self.weights = weights
        for i, modes in enumerate(modalities):
            if distributions[i] == 'P':
                mode_losses = [PoissonNegativeLikelihood(weights[i])] * len(modes.split('-'))
            elif distributions[i] == 'G':
                dims = [inputs[mode][0].shape[1] for mode in modes.split('-')]
                mode_losses = []
                for j in range(len(dims)):
                    dim_sum = 1 if len(dims) == 1 else sum(dims) - dims[j]
                    mode_losses.append(GaussianNegativeLikelihood(dim_sum=dim_sum, 
                                                                  weight=weights[i],
                                                                  **gaussian_loss_kwargs).to(device))
            else:
                raise NotImplementedError('Specified distribution is not implemented.')

            self.hidden_tensors[modes] = {
                'tensor': HiddenInteractionTensor(modes.split('-'), self.factors).to(device),
                'binary_flags': [inputs[mode][1] for mode in modes.split('-')],
                'loss_funcs': mode_losses
            }

        self.labels = labels
        
        self.angular = PairwiseAngularRegularization(rank, angular_threshold)
        self.angular_weight = angular_weight

        self.elastic_net = ElasticNet(elastic_l1_ratio)
        self.elastic_weight = elastic_weight

        self.writer = tb_writer
    
    def validate_prediction(self):
        lr = LogisticRegressionCV(cv=5, class_weight='balanced', solver='liblinear')
        Xtrain, Xtest, y_train, y_test = train_test_split(self.factors.pt_reps.data.cpu().numpy(), self.labels,
                                                          train_size=0.8, test_size=0.2,
                                                          random_state=19)
        lr.fit(Xtrain, y_train)
        pred_prob = lr.predict_proba(Xtest)[:, 1]
        auc = roc_auc_score(y_test, pred_prob)
        ap = average_precision_score(y_test, pred_prob)
        return auc, ap

    def _get_item(self, x):
        if isinstance(x, torch.Tensor):
            return x.item()
        elif isinstance(x, int):
            return x
        else:
            raise TypeError('data type not supported.')


    def fit(self, lr_init=0.0001, weight_decay=0, max_iters=1000000):
        optimizer = Adam(self.factors.parameters(), lr=lr_init, weight_decay=weight_decay)
        prefix = 'train' if not self.projector else 'projection'
        early_stop = EarlyStopping(patience=800)

        factors = {'pt_reps': self.factors.pt_reps}
        factors.update(self.factors.factors)
        timers = []
        for iter_ in range(max_iters):
            tic = time.time()
            elastic_regs = []
            angular_regs = []
            nll_loss = 0
            total_loss = 0
            for name, X in factors.items():
                if self.projector and name != 'pt_reps':
                    continue  # for projector: update patient representation only

                for _, X_ in factors.items():
                    X_.requires_grad = False
                X.requires_grad = True

                optimizer.zero_grad()
                pt_loss_total = 0
                for modes, tensor_dict in self.hidden_tensors.items():
                    hidden_tensor = tensor_dict['tensor']

                    reconstructions = hidden_tensor()
                    targets = [self.inputs[mode][0] for mode in modes.split('-')]
                    binary_flags = tensor_dict['binary_flags']
                    loss_funcs = tensor_dict['loss_funcs']

                    for V, V_hat, bin_, loss_func in zip(targets, reconstructions, binary_flags, loss_funcs):
                        pt_loss = loss_func(V_hat, V, bin_)
                        pt_loss_total += pt_loss
                # pt_loss_total = pt_loss_total.mean() #/ self.num_pt

                nll_loss = pt_loss_total.mean()
                if name != 'pt_reps':
                    angular_reg = self.angular_weight * self.angular(X)
                    elastic_reg = self.elastic_weight * self.elastic_net(X)
                else:
                    angular_reg = elastic_reg = 0

                total_loss = nll_loss + angular_reg + elastic_reg
                total_loss.backward()
                optimizer.step()
                self.factors.non_negative_projection()
                
                angular_reg = self._get_item(angular_reg)
                elastic_reg = self._get_item(elastic_reg)

                angular_regs.append(angular_reg)
                elastic_regs.append(elastic_reg)

                if self.writer is not None:
                    self.writer.add_scalar(prefix+'/angular_'+name, angular_reg, iter_)
                    self.writer.add_scalar(prefix+'/elastic_'+name, elastic_reg, iter_)

            if self.writer is not None:
                self.writer.add_scalar(prefix+'/total_loss', total_loss.item(), iter_)
                self.writer.add_scalar(prefix+'/nll_loss', nll_loss.item(), iter_)

            timers.append(time.time() - tic)
            if iter_ == 0 or (iter_ + 1) % 500 == 0:
                time_avg = sum(timers) / len(timers)
                timers = []
                
                angular_reg_str = f'{torch.FloatTensor(angular_regs[1:]).mean():.3e}'
                elastic_reg_str = f'{torch.FloatTensor(elastic_regs[1:]).mean():.3e}'
                
                info_str = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] '
                info_str += f'Iteration {iter_+1}: loss={total_loss.item():.3e} | '
                info_str += f'nll={nll_loss.item():.3e} | '
                info_str += f'angular={angular_reg_str} | elastic={elastic_reg_str} | ' if not self.projector else ''
                info_str += f'time={time_avg*1000:.1f}ms/it.'
                print(info_str)

            if early_stop(-total_loss.item(), self.factors):
                break

    def construct_projector(self, test_inputs):
        num_pt_test = test_inputs[list(test_inputs.keys())[0]][0].shape[0]
        rank = self.factors.pt_reps.shape[1]

        factors = deepcopy(self.factors).to(self.device)
        factors.pt_reps = nn.Parameter(torch.rand(num_pt_test, rank))
        for k, v in factors.factors.items():
            v.requires_grad = False

        projector = CollectiveHITF(inputs=test_inputs,
                                   modalities=self.modalities,
                                   distributions=self.distributions,
                                   rank=self.rank,
                                   device=self.device,
                                   gaussian_loss_kwargs=self.gaussian_loss_kwargs,
                                   init_factors=factors,
                                   weights=self.weights,
                                   tb_writer=self.writer,
                                   projector=True)
        return projector

    def save_factors(self, fp):
        torch.save(self.factors.state_dict(), fp)

    def load_factors(self, fp):
        self.factors.load_state_dict(torch.load(fp))




class EarlyStopping(object):
    def __init__(self, patience, restore_best=True, tolerance=None, relative_tolerance=1e-4):
        self.best_score = None
        self.plateau_counter = 0
        self.patience = patience
        self.stopped = False
        self.restore_best = restore_best
        self.best_state_dict = None
        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
    
    def is_improved(self, score):
        if self.tolerance is not None:
            return score - self.best_score >= self.tolerance
        if self.relative_tolerance is not None:
            return score - self.best_score >= self.relative_tolerance * self.best_score
        return score >= self.best_score

    def __call__(self, score, model):
        if self.best_score is None or self.is_improved(score):
            self.plateau_counter = 0
            self.best_score = score
            if self.restore_best:
                self.best_state_dict = deepcopy(model.state_dict())

        else:
            self.plateau_counter += 1
            if self.plateau_counter >= self.patience:
                self.stopped = True
                if self.restore_best:
                    model.load_state_dict(self.best_state_dict)

        return self.stopped

