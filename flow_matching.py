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

@torch.no_grad()
def generate_mnist_images_odeint(model, num_images, device, steps=2, method="rk4"):
    model.eval()

    x0 = torch.randn(num_images, 1, 28, 28, device=device)

    # odeint 的 t 是「時間點列表」，不是 step_size；點越多越細
    # 例如 0->1 切 101 個點，最後取 [-1]
    t = torch.linspace(0.0, 1.0, 101, device=device)

    func = ODEFuncImage(model, batch_size=num_images)
    sol = odeint(func, y0=x0, t=t, method=method)   # sol: [T,B,1,28,28]
    x1 = sol[-1]

    return x1

def generate_graphs(model, num_mols, node_feats, edge_feats, max_nodes, device, name, mu, distribution, tau_sched, loss_function, counter, small_model):
    mols_per_b = num_mols // 10
    all_mols = []

    for _ in tqdm(range(10), desc='Generating molecules'):
        x_params, e_params = torch.zeros(mols_per_b, max_nodes, node_feats), torch.zeros(mols_per_b, max_nodes, max_nodes, edge_feats)

        if distribution == 'normal':
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
            x_0, e_0 = torch.randn(mols_per_b, max_nodes, node_feats), torch.randn(mols_per_b, max_nodes, max_nodes, edge_feats)
            e_0 = 0.5 * (e_0 + e_0.transpose(1, 2))

            x_t, e_t = x_0, e_0
            x_t, e_t = x_t.to(device), e_t.to(device)

            gumbel = None
            tau_t = None

        elif distribution == 'normal_projected':
            M = torch.ones(x_params.shape[-1], device=x_params.device) / x_params.shape[-1]
            samples_x = torch.randn_like(x_params) + M
            x_t = hyperplane_proj(samples_x)

            M = torch.ones(e_params.shape[-1], device=x_params.device) / x_params.shape[-1]
            samples_e = torch.randn_like(e_params) + M
            e_t = hyperplane_proj(samples_e)
            e_t = 0.5 * (e_t + e_t.transpose(1, 2))

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
            torch.cat([torch.ones(sample, dtype=torch.bool), torch.zeros(max_nodes - sample, dtype=torch.bool)]) for sample in samples
        ])
        # PlaceHolder(X=X, E=E, y=None), node_mask
        from utils import PlaceHolder

        result_tensor = result_tensor.to(x_t.device)
        graph = PlaceHolder(X=x_t, E=e_t, y=None)
        graph.mask(result_tensor)
        # graph = graph.mask(mask)
        x_t, e_t = graph.X.float(), graph.E.float()
        if small_model == 1:
            t = torch.tensor([0.0, 1.0])
        else:
            t = torch.tensor([0.0, 0.95])

        t = t.to(device)
        x_flat, e_flat = x_t.flatten().unsqueeze(-1), e_t.flatten().unsqueeze(-1)
        y_in = torch.cat([x_flat, e_flat], dim=0)

        result_tensor = result_tensor.to(device)
        y_in = y_in.to(device)
        cut_off = x_flat.shape[0]

        size_x, size_e = x_t.shape, e_t.shape
        odeNet = ODEFuncGraph(model, size_x, size_e, cut_off, max_nodes, distribution, tau_t, loss_function, mu, gumbel, result_tensor)

        all_all_mols = defaultdict(list)

        for k in [100]:
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

def eval_and_log_molecules(mols, log, smiles, device):
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

        # can_train = [Chem.CanonSmiles(train_smile) for train_smile in train_smiles]
        # can_test = [Chem.CanonSmiles(test_smile) for test_smile in test_smiles]
        # can_gen = [Chem.CanonSmiles(mol) for mol in unique_mols]
        # can_novel = set(can_gen) - set(can_test)
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

# ---- FID ----
def _to_uint8_3ch(x01: torch.Tensor) -> torch.Tensor:
    """
    x01: float tensor in [0,1], shape [B,1,H,W] or [B,3,H,W]
    return: uint8 tensor in [0,255], shape [B,3,H,W]
    """
    if x01.ndim != 4:
        raise ValueError(f"Expect [B,C,H,W], got {x01.shape}")
    if x01.size(1) == 1:
        x01 = x01.repeat(1, 3, 1, 1)
    x01 = x01.clamp(0, 1)
    x255 = (x01 * 255.0).round().to(torch.uint8)
    return x255


@torch.no_grad()
def compute_fid_torchmetrics(
    x_gen_01: torch.Tensor,      # [N,1,28,28] float in [0,1]
    test_loader,
    device,
    max_real: int = 10_000,
    max_fake: int = 10_000,
):
    """
    用 torchmetrics 的 FrechetInceptionDistance 計算 FID。
    注意：torchmetrics 的 FID 用 Inception，通常預期輸入 >= 3x299x299。
    torchmetrics 內部會 resize，但 MNIST 仍可用（只是不如在自然影像上那麼標準）。
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except Exception as e:
        raise ImportError(
            "需要 torchmetrics：pip install torchmetrics"
        ) from e

    fid = FrechetInceptionDistance(feature=2048).to(device)

    # real
    seen = 0
    for x, _ in test_loader:
        x = x.to(device).float()
        # 你資料如果是 Normalize 過的，請先反正規化到 [0,1]
        x01 = x
        x_u8 = _to_uint8_3ch(x01)
        fid.update(x_u8, real=True)
        seen += x_u8.size(0)
        if seen >= max_real:
            break

    # fake
    x_gen_01 = x_gen_01.to(device).float()
    if x_gen_01.size(0) > max_fake:
        x_gen_01 = x_gen_01[:max_fake]
    fid.update(_to_uint8_3ch(x_gen_01), real=False)

    return float(fid.compute().item())


# ---- NLL for CNF via divergence integral ----
def _hutch_divergence(v, x, eps):
    """
    v: vector field v(x) same shape as x
    div v ≈ eps^T J_v eps
    """
    # (v * eps).sum() 的梯度 wrt x = J_v^T eps
    inner = (v * eps).sum()
    grad = torch.autograd.grad(inner, x, create_graph=False, retain_graph=False)[0]
    div_est = (grad * eps).sum(dim=(1,2,3))  # per-sample
    return div_est


class CNFLogProbODE(torch.nn.Module):
    """
    解 (x, logp) 的增廣 ODE：
      dx/dt = v_theta(x,t)
      dlogp/dt = -div v_theta(x,t)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, state):
        x, logp = state  # x: [B,1,28,28], logp: [B]
        x = x.requires_grad_(True)

        t_in = t.view(1,1,1,1).expand(x.size(0),1,1,1)
        v = self.model(x, t_in)

        eps = torch.randn_like(x)
        div = _hutch_divergence(v, x, eps)     # [B]
        dlogp = -div                           # [B]

        return v, dlogp


@torch.no_grad()
def estimate_nll_on_test(model, test_loader, device, max_batches: int = 50, ode_steps: int = 101, method: str = "rk4"):
    """
    估計 test data 的 NLL（平均每張圖的 negative log-likelihood，單位 nats）。
    做法：把 x1(test image) 從 t=1 反向積分到 t=0 得到 x0，並積分 divergence。
      log p(x1) = log p0(x0) - ∫_0^1 div v dt
    反向積分：t: 1 -> 0
    """
    from torchdiffeq import odeint

    model.eval()
    ode_func = CNFLogProbODE(model).to(device)

    # 反向時間
    t = torch.linspace(1.0, 0.0, ode_steps, device=device)

    total_nll = 0.0
    total_n = 0

    for bi, (x1, _) in enumerate(test_loader):
        if bi >= max_batches:
            break

        x1 = x1.to(device).float()
        B = x1.size(0)

        # 如果你的 test x1 不是在模型生成空間（例如 Normalize 過），這裡要一致
        # x1_model = (x1 - mean)/std 或 x1_model = x1*2-1 等
        x1_model = x1

        logp1_init = torch.zeros(B, device=device)

        # 需要梯度來算 divergence，所以這段不能用 @torch.no_grad()
        # 我們在外層 no_grad 了，這裡局部開啟 grad：
        with torch.enable_grad():
            x0, logp0 = odeint(ode_func, (x1_model, logp1_init), t, method=method)
            x0 = x0[-1]      # [B,1,28,28]
            logp0 = logp0[-1]  # [B]

            # base log p0 for standard Normal
            # log N(x0;0,I) = -0.5*(||x0||^2 + D*log(2pi))
            D = x0[0].numel()
            log_base = -0.5 * (x0.view(B, -1).pow(2).sum(dim=1) + D * torch.log(torch.tensor(2.0 * torch.pi, device=device)))

            # log p(x1) = log_base + logp0  (因為 logp 在 ODE 裡累積的是 -∫div)
            logp_x1 = log_base + logp0
            nll = -logp_x1  # [B]

        total_nll += float(nll.sum().item())
        total_n += B

    return total_nll / max(total_n, 1)

def eval_and_log_images(gen_images, model, test_loader,
                        log=False, device="cuda", wandb_run=None, # 你若用 wandb，可傳 wandb
                        max_real=10_000, max_fake=10_000, nll_max_batches=50):
    """
    # gen_images: dict: k -> generated images tensor [N,1,28,28] in [0,1] (建議)
    回傳：results dict, 以及每個 k 對應的 fid/nll
    """
    results = {}
    # NLL 通常跟 k 無關（NLL 是 model 對 test 的 likelihood），所以算一次即可
    nll = estimate_nll_on_test(model, test_loader, device=device, max_batches=nll_max_batches)
    print(f"[NLL] test NLL (nats/image): {nll:.4f}")

    
    fid = compute_fid_torchmetrics(x_gen_01=gen_images, test_loader=test_loader, device=device, max_real=max_real, max_fake=max_fake)
    print(f"FID: {fid:.4f}, NLL(test): {nll:.4f}")

    if log and wandb_run is not None:
        wandb_run.log({
            f"FID": fid,
            "NLL_test": nll,
        })

    results = {"FID": fid, "NLL_test": nll}

    return results

class ODEFuncImage(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size

    def forward(self, t_scalar, x):
        # t_scalar: scalar tensor from odeint
        t = t_scalar.view(1,1,1,1).expand(self.batch_size,1,1,1)
        return self.model(x.float(), t.float())

class ODEFuncGraph(torch.nn.Module):
    def __init__(self, neural_network, size_x, size_e, cut_off, max_nodes, distribution, tau_t, loss_function, mu, gumbel, mask):
        super(ODEFuncGraph, self).__init__()
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
        else:
            x_1, e_1 = pred.X, pred.E
            x_1, e_1 = torch.softmax(x_1, dim=-1), torch.softmax(e_1, dim=-1)

            v_x = (x_1 - x_t) / (1 - t)
            v_e = (e_1 - e_t) / (1 - t)

        v = torch.cat([v_x.flatten().unsqueeze(-1), v_e.flatten().unsqueeze(-1)], dim=0)
        return v
