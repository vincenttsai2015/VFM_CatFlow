import torch
import torch
import pandas as pd
import numpy as np
import os
from utils import make_molecule, project_simplex
import matplotlib.pyplot as plt
# import default dict
from collections import defaultdict
import torch.nn as nn
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import QM9, ZINC

from itertools import product
from torch_geometric.loader import DataLoader as GraphDataLoader
import wandb
from tqdm import tqdm
from utils import get_tau_sched
import torch_geometric.utils
from torch_geometric.utils import sort_edge_index, to_undirected, to_dense_adj, to_dense_batch
from math import sqrt
from structural import add_feats



def conditional_velocity(distribution, x_1, t, tau_t, mu, sigma, edge=False):
    if distribution == 'normal':
        x_0 = torch.randn_like(x_1)
        x_0 = 0.5 * (x_0 + x_0.transpose(1, 2)) if edge else x_0
        diag = torch.eye(x_1.shape[1], dtype=torch.bool).unsqueeze(0).expand(x_1.shape[0], -1, -1)
        x_t = (1 - t) * x_0 + t * x_1

        v_x = x_1 - x_0

        noise = torch.randn_like(x_t) * 0.5
        noise = 0.5 * (noise + noise.transpose(1, 2)) if edge else noise

        x_t = x_t + noise

    elif distribution == 'simplex':
        # sample gumbel
        pi_0 = 1/4 * torch.ones_like(x_1)
        x_0 = torch.ones_like(x_1)
        x_0 = torch.distributions.dirichlet.Dirichlet(x_0)
        x_0 = x_0.sample()

        gumbel_x = -torch.log(-torch.log(torch.rand_like(x_1.float()) + 1e-10) + 1e-10)
        gumbel_x = 0.5 * (gumbel_x + gumbel_x.transpose(1, 2)) if edge else gumbel_x
        f = lambda t: torch.softmax((torch.log((1-t) * x_0 + t * x_1) + gumbel_x) / tau_t(t), dim=-1)
        x_t, v_x = torch.autograd.functional.jvp(f, t, torch.ones_like(t))
    
    elif distribution == 'normal_simplex':
        x_0 = torch.randn_like(x_1)
        v_x = x_1 - x_0
        x_t = t * x_1 + (1 - t) * x_0

        noise = torch.randn_like(x_t) * 1e-6
        noise = 0.5 * (noise + noise.transpose(1, 2)) if edge else noise
        x_t += noise

    elif distribution == 'normal_projected':
        M = torch.ones(x_1.shape[-1], device=x_1.device) / x_1.shape[-1]

        # Sample points around M for all samples at once
        samples = torch.randn_like(x_1) + M

        x_0 = hyperplane_proj(samples)
        x_0 = 0.5 * (x_0 + x_0.transpose(1, 2)) if edge else samples

        x_t = t * x_1 + (1 - t) * x_0
        v_x = x_1 - x_0

    return x_t, v_x


def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)



# def hyperplane_proj(seq):
#     """
#     Project points onto H^n by ensuring the sum of elements equals 1.
#     """
#     Y = seq.reshape(-1, seq.shape[-1])
#     N, K = Y.shape
#     # Calculate the sum of elements for each vector and the correction needed
#     total_sum = torch.sum(Y, dim=-1, keepdim=True)
#     correction = (1 - total_sum) / K
#     # Apply correction to each element
#     X = Y + correction
#     return X.view(seq.shape)

def hyperplane_proj(seq):
    import torch
    """
    Orthogonally project points onto the hyperplane sum(x_i) = 1.
    """
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    # Normal vector of the hyperplane
    n = torch.ones(K)
    n = n.to(Y.device)

    n_dot_n = torch.dot(n, n)
    # Calculate the dot product of each vector with the normal vector
    n_dot_Y = torch.matmul(Y, n)
    # Calculate the correction for each vector
    correction = (1 - n_dot_Y) / n_dot_n
    # Apply correction to each element
    X = Y + torch.ger(correction, n)
    return X.view(seq.shape)


def generate_graphs(model, num_mols, node_feats, edge_feats, max_nodes, device, name, mu, distribution, tau_sched, loss_function, counter, small_model):
    mols_per_b = num_mols // 10
    all_mols = []

    for _ in tqdm(range(10), desc='Generating molecules'):
        x_params, e_params = torch.zeros(mols_per_b, max_nodes, node_feats), torch.zeros(mols_per_b, max_nodes, max_nodes, edge_feats)

        if distribution == 'normal':
            # x_0 = torch.randn_like(x_1)
            # x_0 = 0.5 * (x_0 + x_0.transpose(1, 2)) if edge else x_0
            # x_t = (1 - t) * x_0 + t * x_1
            # v_x = x_1 - x_0

            x_t = torch.randn_like(x_params)
            e_t = torch.randn_like(e_params)
            e_t = (e_t + torch.transpose(e_t, 1, 2)) / 2

            noise_x = torch.randn_like(x_t) * 1e-6
            noise_e = torch.randn_like(e_t) * 1e-6
            noise_e = (noise_e + torch.transpose(noise_e, 1, 2)) / 2

            x_t = x_t + noise_x
            e_t = e_t + noise_e

            x_t, e_t = x_t.to(device), e_t.to(device)

            gumbel = None
            tau_t = None 
        elif distribution == 'simplex':
            gumbel_x = -torch.log(-torch.log(torch.rand_like(x_params.float()) + 1e-10) + 1e-10)
            gumbel_e = -torch.log(-torch.log(torch.rand_like(e_params.float()) + 1e-10) + 1e-10)
            gumbel_e = 0.5 * (gumbel_e + gumbel_e.transpose(1, 2))

            tau_t = get_tau_sched(tau_sched)

            gumbel_x, gumbel_e = gumbel_x.to(device), gumbel_e.to(device)
            x_0, e_0 = (1/4) * torch.ones_like(x_params).to(device), (1/4) * torch.ones_like(e_params).to(device)
            # sample from dirichlet
            x_0, e_0 = torch.ones_like(x_params), torch.ones_like(e_params)
            x_0, e_0 = torch.distributions.dirichlet.Dirichlet(x_0), torch.distributions.dirichlet.Dirichlet(e_0)
            x_0, e_0 = x_0.sample(), e_0.sample()

            x_0, e_0 = x_0.to(device), e_0.to(device)

            x_t = torch.softmax((torch.log(x_0) + gumbel_x) / tau_t(torch.zeros(1).to(device)), dim=-1)
            e_t = torch.softmax((torch.log(e_0) + gumbel_e) / tau_t(torch.zeros(1).to(device)), dim=-1)

            gumbel = (gumbel_x, gumbel_e)

        elif distribution == 'normal_simplex':
            # x = torch.randn_like(x_params)
            # shift = torch.tensor([1 / 4] * 4).reshape(1, 1, 4)  # Shape this to broadcast over B and N
            # y = x + shift
            #
            # normal_vector = torch.tensor([1., 1., 1., 1.], device=y.device, dtype=y.dtype)
            # normal_vector = normal_vector / normal_vector.norm()  # Normalize
            #
            # # Compute the dot product along K dimension
            # dot_product = torch.sum(y * normal_vector, dim=-1, keepdim=True)
            #
            # # Compute projection onto the normal vector
            # projection = y - (dot_product / 4.0) * normal_vector
            #
            # # Adjustment to satisfy the plane equation, considering the shift for the equation = 1
            # adjustment = (1 - torch.sum(projection, dim=-1, keepdim=True)) / 4
            # x_0 = projection + adjustment

            #
            # n = torch.ones(x.shape[-1], device=x.device) / torch.sqrt(
            #     torch.tensor(x.shape[-1], device=x.device, dtype=x.dtype))
            #
            # x_dot_n = torch.sum(x * n, dim=-1, keepdim=True)
            # n_norm_squared = torch.sum(n * n)
            # d = 1 / torch.sqrt(torch.tensor(x.shape[-1], device=x.device, dtype=x.dtype))
            # projection = x - ((x_dot_n - d) / n_norm_squared) * n
            #
            # x_0 = projection

            # x = torch.randn_like(e_params)
            # shift = torch.tensor([1 / 4] * 4).reshape(1, 1, 4)  # Shape this to broadcast over B and N
            # y = x + shift
            #
            # normal_vector = torch.tensor([1., 1., 1., 1.], device=y.device, dtype=y.dtype)
            # normal_vector = normal_vector / normal_vector.norm()  # Normalize
            #
            # # Compute the dot product along K dimension
            # dot_product = torch.sum(y * normal_vector, dim=-1, keepdim=True)
            #
            # # Compute projection onto the normal vector
            # projection = y - (dot_product / 4.0) * normal_vector
            #
            # # Adjustment to satisfy the plane equation, considering the shift for the equation = 1
            # adjustment = (1 - torch.sum(projection, dim=-1, keepdim=True)) / 4
            # e_0 = projection + adjustment

            #
            # n = torch.ones(x.shape[-1], device=x.device) / torch.sqrt(
            #     torch.tensor(x.shape[-1], device=x.device, dtype=x.dtype))
            #
            # x_dot_n = torch.sum(x * n, dim=-1, keepdim=True)
            # n_norm_squared = torch.sum(n * n)
            # d = 1 / torch.sqrt(torch.tensor(x.shape[-1], device=x.device, dtype=x.dtype))
            # projection = x - ((x_dot_n - d) / n_norm_squared) * n
            #
            # x_0 = projection

            x_0, e_0 = torch.randn(mols_per_b, max_nodes, node_feats), torch.randn(mols_per_b, max_nodes, max_nodes, edge_feats)
            e_0 = 0.5 * (e_0 + e_0.transpose(1, 2))

            x_t, e_t = x_0, e_0
            x_t, e_t = x_t.to(device), e_t.to(device)

            gumbel = None
            tau_t = None

        elif distribution == 'normal_projected':
            # M = torch.ones(x_params.shape[-1], device=x_params.device) / x_params.shape[-1]
            # x_t = torch.randn_like(x_params) + M
            # e_t = torch.randn_like(e_params) + M
            # e_t = 0.5 * (e_t + e_t.transpose(1, 2))
            # x_t, e_t = hyperplane_proj(x_t), hyperplane_proj(e_t)
            # x_t, e_t = x_t.to(device), e_t.to(device)

            # M = torch.ones(x_1.shape[-1], device=x_1.device) / x_1.shape[-1]
            #
            # # Sample points around M for all samples at once
            # samples = torch.randn_like(x_1) + M
            # samples = 0.5 * (samples + samples.transpose(1, 2)) if edge else samples
            #
            # x_0 = hyperplane_proj(samples)

            M = torch.ones(x_params.shape[-1], device=x_params.device) / x_params.shape[-1]
            samples_x = torch.randn_like(x_params) + M
            x_t = hyperplane_proj(samples_x)
            # noise = torch.randn_like(x_0) * 1e-6
            # noise = hyperplane_proj(noise)
            # x_t = x_0 + noise

            M = torch.ones(e_params.shape[-1], device=x_params.device) / x_params.shape[-1]
            samples_e = torch.randn_like(e_params) + M
            e_t = hyperplane_proj(samples_e)
            e_t = 0.5 * (e_t + e_t.transpose(1, 2))

            # noise = torch.randn_like(e_0) * 1e-6
            # noise = 0.5 * (noise + noise.transpose(1, 2))
            # noise = hyperplane_proj(noise)
            #
            # e_t = e_0 + noise

            x_t, e_t = x_t.to(device), e_t.to(device)

            gumbel = None
            tau_t = None

        x_t = x_t.to(device)
        e_t = e_t.to(device)

        # sample size

        probs = counter / counter.sum()
        # torch version of: samples = np.random.choice(np.arange(1, max_nodes + 2), p=probs, size=x_params.shape[0])
        samples = torch.multinomial(probs, x_params.shape[0], replacement=True)

        result_tensor = torch.stack([
            torch.cat([torch.ones(sample, dtype=torch.bool), torch.zeros(max_nodes - sample, dtype=torch.bool)])
            for sample in samples
        ])
        # PlaceHolder(X=X, E=E, y=None), node_mask
        from utils import PlaceHolder

        result_tensor = result_tensor.to(x_t.device)
        graph = PlaceHolder(X=x_t, E=e_t, y=None)
        graph.mask(result_tensor)
        # graph = graph.mask(mask)
        x_t, e_t = graph.X.float(), graph.E.float()
        # else:
        #     result_tensor = None
        #     result_tensor = torch.ones(size=(x_t.shape[0], max_nodes)).bool().to(x_t.device)
        # t = torch.tensor([0.05, 0.95])
        if small_model == 1:
            t = torch.tensor([0.0, 1.0])
        else:
            t = torch.tensor([0.0, 0.95])
        # t = torch.linspace(0.0, 1.0, 5)v


        t = t.to(device)
        x_flat, e_flat = x_t.flatten().unsqueeze(-1), e_t.flatten().unsqueeze(-1)
        y_in = torch.cat([x_flat, e_flat], dim=0)

        result_tensor = result_tensor.to(device)
        y_in = y_in.to(device)
        cut_off = x_flat.shape[0]

        size_x, size_e = x_t.shape, e_t.shape
        odeNet = ODEFunc(model, size_x, size_e, cut_off, max_nodes, distribution, tau_t, loss_function, mu, gumbel, result_tensor)
        # solution = odeint(odeNet, y0=y_in, t=t)
        # solution = odeint(odeNet, y0=y_in, t=t, method='euler', options={'step_size': 0.01})[-1]

        # solution = odeint(odeNet, y0=y_in, t=t, method='euler', options={'step_size': 0.01})[-1]

        # solution = odeint(odeNet, y0=y_in, t=t, method='euler', options={'step_size': (1/100)})[-1]

        # solution = odeint(odeNet, y0=y_in, t=t, atol=1e-5, rtol=1e-5)[-1]

        all_all_mols = defaultdict(list)

        for k in [100]:
            # solution = odeint(odeNet, y0=y_in, t=t, method='euler', options={'step_size': (1/k)})[-1]
            if small_model == 1:
                solution = odeint(odeNet, y0=y_in, t=t, method='euler', options={'step_size': (1/k)})[-1]
            else:
                solution = odeint(odeNet, y0=y_in, t=t, atol=1e-5, rtol=1e-5)[-1]

            x_gen, e_gen = solution[:x_flat.shape[0]], solution[x_flat.shape[0]:]
            x_gen = x_gen.reshape(x_t.shape)
            e_gen = e_gen.reshape(e_t.shape)

            mols = [make_molecule(x, e, max_nodes, type='guacamol') for x, e in zip(x_gen, e_gen)]
            all_all_mols[k].extend(mols)

    return all_all_mols

def eval_and_log(mols, log, smiles, device):
    all_valid_mols, all_unique_mols, all_novel_mols = defaultdict(list), defaultdict(list), defaultdict(list)

    for k, k_mols in mols.items():

        valid_mols, unique_mols, novel_mols = [], [], []
        valid, unique, novel = 0, 0, 0

        for mol in k_mols:
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                valid_mols.append(largest_mol)
                valid += 1
            except:
                continue

        train_smiles, test_smiles = smiles
        unique_set = set([Chem.MolToSmiles(mol) for mol in valid_mols])
        novel_set = unique_set - set(train_smiles)

        percentage_valid = valid / len(k_mols)
        percentage_unique = len(unique_set) / len(valid_mols) if len(valid_mols) > 0 else 0
        percetnage_novel = len(novel_set) / len(unique_set) if len(unique_set) > 0 else 0
        print(f'For k={k}, Valid: {percentage_valid}, Unique: {percentage_unique}, Novel: {percetnage_novel}')

        can_train = [Chem.CanonSmiles(train_smile) for train_smile in train_smiles]
        can_test = [Chem.CanonSmiles(test_smile) for test_smile in test_smiles]
        can_gen = [Chem.CanonSmiles(mol) for mol in unique_mols]
        can_novel = set(can_gen) - set(can_test)
        unique_list = list(unique_set)
        # test_list = list(test_smiles)

        if len(unique_list) > 0:
            from fcd import get_fcd, load_ref_model, canonical_smiles, get_predictions, calculate_frechet_distance
            model = load_ref_model()
            can_test = [w for w in canonical_smiles(test_smiles) if w is not None]
            can_gen = [w for w in canonical_smiles(unique_list) if w is not None]
            can_train = [w for w in canonical_smiles(train_smiles) if w is not None]

            # get novel ones
            can_novel = set(can_gen) - set(can_train)
            novelty = len(can_novel) / len(can_gen)

            test_fc = get_predictions(model, can_test)
            gen_fc = get_predictions(model, can_gen)

            def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
                import numpy as np
                from scipy.linalg import sqrtm
                """Calculate the Frechet distance between two multivariate Gaussians."""
                mu_diff = mu1 - mu2

                # Add a small epsilon to the covariance matrices to ensure positive semi-definiteness
                sigma1 += np.eye(sigma1.shape[0]) * eps
                sigma2 += np.eye(sigma2.shape[0]) * eps

                # Compute the square root of the product of covariance matrices
                covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
                if np.iscomplexobj(covmean):
                    covmean = covmean.real

                # Calculate the Fréchet distance
                trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)
                return (mu_diff.dot(mu_diff) + trace_term).real

            # Example usage
            mu_real, sigma_real = np.mean(test_fc, axis=0), np.cov(test_fc, rowvar=False)
            mu_gen, sigma_gen = np.mean(gen_fc, axis=0), np.cov(gen_fc, rowvar=False)
            try:
                fcd_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
            except:
                fcd_score = 1e8
        else:
            novelty = 0
            fcd_score = 1e8
            can_gen = []
            can_novel = []

        # if len(unique_list) > 0:
        #     try:
        #         fcd = FCD(device=device)
        #         print(fcd(test_list, unique_list))
        #     except:
        #         print("Couldn't compute")
        #         pass
        # else:
        #     print('No valid molecules')

        # can_novel = len(can_novel) / len(can_gen) if len(can_gen) > 0 else 0
        # print(f'Validity: ', percentage_valid)
        # print(f'Uniqueness: ', percentage_unique)
        # print('Percentage Novel: ', percetnage_novel)

        print(f'Percentage Valid: {percentage_valid}, Percentage Unique: {percentage_unique}, Novelty: {novelty}, FCD Score: {fcd_score}')

        s = max(percentage_unique, percentage_valid)

        print(f'Percentage Valid: {percentage_valid}, Percentage Unique: {percentage_unique}, Ablation Score: {s}')

        if log:
            wandb.log({
                f'Validity': percentage_valid,
                f'Uniqueness': percentage_unique,
                f'Novelty': novelty,
                f'FCD': fcd_score,
                f'Ablation': s
            })

        all_valid_mols[k] = valid_mols

    return all_valid_mols
# class ODEFunc(torch.nn.Module):
#     def __init__(self, neural_network, size_x, size_e, cut_off, max_nodes, distribution, tau_t, loss_function, mu, gumbel):
#         super(ODEFunc, self).__init__()
#         self.neural_network = neural_network
#         self.size_x = size_x
#         self.size_e = size_e
#         self.cut_off = cut_off
#         self.max_nodes = max_nodes
#         self.distribution = distribution
#         self.tau_t = tau_t
#         self.loss_function = loss_function
#         self.mu = mu
#         self.gumbel = gumbel
#
#     def forward(self, t, y):
#
#
#         x, e = y[:self.cut_off], y[self.cut_off:]
#         x_t = x.reshape(self.size_x)
#         e_t = e.reshape(self.size_e)
#         diag = torch.eye(x_t.shape[1], dtype=torch.bool).unsqueeze(0).expand(x_t.shape[0], -1, -1)
#
#         e_t[diag] = 0
#         assert not torch.isnan(e_t).any()
#         y_t = torch.ones(size=(x_t.shape[0], 1)).to(x_t.device) * t
#         mask = torch.ones(size=(x_t.shape[0], self.max_nodes)).bool().to(x_t.device)
#
#         e_t_r, x_t_r = torch.argmax(e_t, dim=-1), torch.argmax(x_t, dim=-1)
#         # e_t_r, x_t_r = torch.nn.functional.one_hot(e_t_r, e_t.size(-1)), torch.nn.functional.one_hot(x_t_r, x_t.size(-1))
#
#         k = t.repeat(x_t.shape[0], 1, 1, 1)
#         # x_t_in, y_t = add_feats(x_t, e_t_r, k, mask, t.device)
#         pred = self.neural_network(x_t, e_t, y_t, mask)
#
#         print(t)
#         if self.loss_function == 'mse':
#             v_x, v_e = pred.X, pred.E
#         else:
#             x_1, e_1 = pred.X, pred.E
#             x_1, e_1 = torch.softmax(x_1, dim=-1), torch.softmax(e_1, dim=-1)
#             # if t < 0 or t > 1:
#             #     v_x, v_e = torch.zeros_like(x_1), torch.zeros_like(e_1)
#             # else:
#
#             if self.distribution == 'normal' or self.distribution == 'normal_simplex':
#                     v_x = (x_1 - x_t) / (1 - t)
#                     v_e = (e_1 - e_t) / (1 - t)
#
#             if self.distribution == 'simplex':
#                 x_0, e_0 = (1 / 4) * torch.ones_like(x_1), (1 / 4) * torch.ones_like(e_1)
#                 gumbel_x = torch.log(x_t / ((1-t) * x_0 + t * x_1) + 1e-8)
#                 gumbel_e = torch.log(e_t / ((1-t) * e_0 + t * e_1) + 1e-8)
#
#                 f_x = lambda t: torch.softmax((torch.log(t * x_1 + (1-t) * x_0 + 1e-8) + gumbel_x) / 1, dim=-1)
#                 _, v_x = torch.autograd.functional.jvp(f_x, t, torch.ones_like(t))
#
#                 f_e = lambda t: torch.softmax((torch.log(t * e_1 + (1-t) * e_0 + 1e-8) + gumbel_e), dim=-1)
#                 _, v_e = torch.autograd.functional.jvp(f_e, t, torch.ones_like(t))
#
#
#         # e_t[diag], v_e[diag] = 0, 0
#
#         v = torch.cat([v_x.flatten().unsqueeze(-1), v_e.flatten().unsqueeze(-1)], dim=0)
#         return v


class ODEFunc(torch.nn.Module):
    def __init__(self, neural_network, size_x, size_e, cut_off, max_nodes, distribution, tau_t, loss_function, mu, gumbel, mask):
        super(ODEFunc, self).__init__()
        self.neural_network = neural_network
        self.size_x = size_x
        self.size_e = size_e
        self.cut_off = cut_off
        self.max_nodes = max_nodes
        self.distribution = distribution
        self.tau_t = tau_t
        self.loss_function = loss_function
        self.mu = mu
        self.gumbel = gumbel
        self.mask = mask

    def forward(self, t, y):
        x, e = y[:self.cut_off], y[self.cut_off:]
        x_t = x.reshape(self.size_x)
        e_t = e.reshape(self.size_e)
        diag = torch.eye(x_t.shape[1], dtype=torch.bool).unsqueeze(0).expand(x_t.shape[0], -1, -1)

        e_t[diag] = 0

        # print(e_t, t)
        assert not torch.isnan(e_t).any()
        y_t = torch.ones(size=(x_t.shape[0], 1)).to(x_t.device) * t
        # mask = torch.ones(size=(x_t.shape[0], self.max_nodes)).bool().to(x_t.device)

        mask = self.mask

        mask = mask.to(x_t.device)
        e_t = e_t.to(x_t.device)
        y_t = y_t.to(x_t.device)
        pred = self.neural_network(x_t, e_t, y_t, mask)

        if self.loss_function == 'mse':
            v_x, v_e = pred.X, pred.E
            # x_1, e_1 = pred.X, pred.E
            # v_x = (x_1 - x_t) / (1 - t)
            # v_e = (e_1 - e_t) / (1 - t)
        else:
            x_1, e_1 = pred.X, pred.E
            x_1, e_1 = torch.softmax(x_1, dim=-1), torch.softmax(e_1, dim=-1)

            v_x = (x_1 - x_t) / (1 - t)
            v_e = (e_1 - e_t) / (1 - t)

            # if self.distribution == 'normal' or self.distribution == 'normal_projected':
            #     v_x = (x_1 - x_t) / (1 - t)
            #     v_e = (e_1 - e_t) / (1 - t)
            # if self.distribution == 'simplex':
            #     x_0, e_0 = (1 / 4) * torch.ones_like(x_1), (1 / 4) * torch.ones_like(e_1)
            #     gumbel_x = torch.log(x_t / ((1-t) * x_0 + t * x_1) + 1e-8)
            #     gumbel_e = torch.log(e_t / ((1-t) * e_0 + t * e_1) + 1e-8)
            #
            #     f_x = lambda t: torch.softmax((torch.log(t * x_1 + (1-t) * x_0 + 1e-8) + gumbel_x) / 1, dim=-1)
            #     _, v_x = torch.autograd.functional.jvp(f_x, t, torch.ones_like(t))
            #
            #     f_e = lambda t: torch.softmax((torch.log(t * e_1 + (1-t) * e_0 + 1e-8) + gumbel_e), dim=-1)
            #     _, v_e = torch.autograd.functional.jvp(f_e, t, torch.ones_like(t))

        # e_t[diag], v_e[diag]  = 0, 0

        v = torch.cat([v_x.flatten().unsqueeze(-1), v_e.flatten().unsqueeze(-1)], dim=0)
        return v
