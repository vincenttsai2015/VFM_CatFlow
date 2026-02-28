import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import QM9, ZINC
from torch_geometric.utils import sort_edge_index, to_undirected, remove_self_loops, to_dense_adj, to_dense_batch
from itertools import product
from torch_geometric.loader import DataLoader as GraphDataLoader
import wandb
from tqdm import tqdm
from math import sqrt


def make_molecule(x, e, size, type):

    dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'Br': 4, 'Cl': 5, 'I': 6, 'P': 7, 'S': 8}
    atom_dict = {v: k for k, v in dict.items()}
    #
    # if type == 'qm9':
    #     atom_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F'}

    molecule = Chem.RWMol()
    for i in range(size):
        atom = torch.argmax(x[i]).item()
        atom = Chem.Atom(atom_dict[atom])
        molecule.AddAtom(atom)

    bonds = torch.argmax(e, dim=-1)

    for i, j in product(range(size), range(size)):
        if i < j:
            bond = bonds[i, j]
            if bond == 4:
                molecule.AddBond(i, j, Chem.BondType.AROMATIC)
            if bond == 3:
                molecule.AddBond(i, j, Chem.BondType.TRIPLE)
            elif bond == 2:
                molecule.AddBond(i, j, Chem.BondType.DOUBLE)
            elif bond == 1:
                molecule.AddBond(i, j, Chem.BondType.SINGLE)

    return molecule


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_dense(x, edge_index, edge_attr, batch, max_nodes):
    x = x.long()
    edge_index = edge_index.long()
    edge_attr = edge_attr.long()

    X, node_mask = to_dense_batch(x=x, batch=batch, max_num_nodes=max_nodes)


    # node_mask = node_mask.float()
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_nodes)
    # E = encode_no_edge(E)

    m = E.sum(dim=3) == 0
    ten = torch.zeros(E.shape[-1], device=E.device)
    ten[0] = 1

    E[m] = ten.long()

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=None), node_mask


def get_loaders(args):
    if args.task == 'qm9_wo_H' or args.task == 'qm9':
        max_nodes = 9
        edge_feats = 4
        node_feats = 4

        if os.path.exists(f'data/{args.task}_train_graphs_{args.small_data}.pt'):
            train_graphs = torch.load(f'data/{args.task}_train_graphs_{args.small_data}.pt')
            val_graphs = torch.load(f'data/{args.task}_val_graphs_{args.small_data}.pt')
            test_graphs = torch.load(f'data/{args.task}_test_graphs_{args.small_data}.pt')
        else:
            train_dataset, val_dataset, test_dataset = generate_loader(args.task, max_nodes, args.batch_size,
                                                                       args.small_data)
            train_graphs, train_smiles = process_graphs(train_dataset, max_nodes)
            val_graphs, val_smiles = process_graphs(val_dataset, max_nodes)
            test_graphs, test_smiles = process_graphs(test_dataset, max_nodes)

            torch.save(train_graphs, f'data/{args.task}_train_graphs_{args.small_data}.pt')
            torch.save(val_graphs, f'data/{args.task}_val_graphs_{args.small_data}.pt')
            torch.save(test_graphs, f'data/{args.task}_test_graphs_{args.small_data}.pt')

        train_loader = GraphDataLoader(train_graphs[:int(args.data_size * len(train_graphs))], batch_size=args.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs[:int(args.data_size * len(val_graphs))], batch_size=args.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
        # get subset of loader

        # save dataloader

    elif args.task == 'zinc':

        max_nodes = 38
        edge_feats = 4
        node_feats = 9
        if os.path.exists(f'data/{args.task}_train_graphs_{args.small_data}.pt'):
            train_graphs = torch.load(f'data/{args.task}_train_graphs_{args.small_data}.pt')
            val_graphs = torch.load(f'data/{args.task}_val_graphs_{args.small_data}.pt')
            test_graphs = torch.load(f'data/{args.task}_test_graphs_{args.small_data}.pt')
        else:
            from torch_geometric.datasets import ZINC
            train_dataset, val_dataset, test_dataset = generate_loader(args.task, max_nodes=38, batch_size=args.batch_size, small_data=args.small_data)
            train_graphs, train_smiles = process_graphs(train_dataset, 38)
            val_graphs, val_smiles = process_graphs(val_dataset, 38)
            test_graphs, test_smiles = process_graphs(test_dataset, 38)

            torch.save(train_graphs, f'data/{args.task}_train_graphs_{args.small_data}.pt')
            torch.save(val_graphs, f'data/{args.task}_val_graphs_{args.small_data}.pt')
            torch.save(test_graphs, f'data/{args.task}_test_graphs_{args.small_data}.pt')

        train_loader = GraphDataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = GraphDataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
        test_loader = GraphDataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    elif args.task == 'cifar10':
        # get cifar 10
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', train=True, download=True),
                                                    batch_size=args.batch_size, shuffle=True)

        for a in train_loader:
            print(a)
            quit()

    elif args.task == 'mnist':
        # get mnist
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor()])
        # get loaders with transform
        train_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=False, transform=transform),
                                                    batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=False, transform=transform),
                                                    batch_size=args.batch_size, shuffle=False, drop_last=True)

        test_loader = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=False, transform=transform),
                                                    batch_size=args.batch_size, shuffle=False, drop_last=True)
        #
        #
        # for x_1, _ in train_loader:
        #     # visualize image
        #     # import Image
        #     import numpy as np
        #     from PIL import Image
        #
        #     img = x_1[0].squeeze().numpy()
        #     img = img * 255
        #     img = img.astype(np.uint8)
        #     img = Image.fromarray(img)
        #     img.show()
        #
        #
        #     quit()

        node_feats = 1
        edge_feats = 1
        max_nodes = 28 * 28

    else:
        raise ValueError(f'Invalid task: {args.task}')


    return train_loader, val_loader, test_loader, node_feats, edge_feats, max_nodes


def process_graphs(dataset, max_nodes):
    all_graphs, smiles = [], []
    loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    c = 0

    for i, graph in tqdm(enumerate(loader)):
        if graph.x.shape[0] == 1:
            continue

        graphs, mask = to_dense(graph.x, graph.edge_index, graph.edge_attr, graph.batch, max_nodes)

        mols = [make_molecule(x, e, max_nodes, None) for x, e in zip(graphs.X, graphs.E)]
        # get largest fragment
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
                Chem.Kekulize(mol)

                # visualize mol
                # img = Draw.MolToImage(mol)
                # img.show()
                # import Draw
                from rdkit.Chem import Draw

                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                # img = Draw.MolToImage(largest_mol)
                # img.show()

                smile = Chem.MolToSmiles(largest_mol)
                smiles.append(smile)
                all_graphs.append(graph)

            except:
                continue

    return all_graphs, smiles


class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2

            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def get_tau_sched(tau_sched, tau_max=5, tau_min=0.1):
    if tau_sched == 'constant':
        tau_t = lambda t: torch.ones_like(t)
    elif tau_sched == 'linear':
        tau_t = lambda t: 1 - t
    elif tau_sched == 'cosine':
        tau_t = lambda t: 0.5 * (1 + torch.cos(t * 3.14159))
    else:
        raise ValueError(f'Invalid tau_sched: {tau_sched}')

    tau_rem = lambda t: (tau_max - tau_min) * tau_t(t) + tau_min

    return tau_rem

def get_CNN_model():
    from models.cnn import ConvNet
    model = ConvNet(in_channels=1, ngf=64)
    return model

def get_GT_model(args, node_feats, edge_feats):
    from models.transformer import GraphTransformer

    if args.small_model == 1:
        hidden_dims = {'dx': 16, 'de': 8, 'dy': 8, 'n_head': 2, 'dim_ffX': 16, 'dim_ffE': 8, 'dim_ffy': 8}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 16}
    elif args.task == 'abstract':
        hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        hidden_mlp_dims = {'X': 128, 'E': 64, 'y': 128}
    else:
        # old
        # hidden_dims = {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        # hidden_mlp_dims = {'X': 128, 'E': 64, 'y': 128}
        hidden_dims = {'dx': 128, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}


    model = GraphTransformer(
        input_dims={'X': node_feats, 'E': edge_feats, 'y': 1},
        # input_dims={'X': node_feats + 3, 'E': edge_feats, 'y': 1 + 6},
        hidden_dims=hidden_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        output_dims={'X': node_feats, 'E': edge_feats, 'y': 1},
        # output_dims={'X': node_feats - 1, 'E': edge_feats, 'y': 1},
        n_layers=args.num_layers,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
    )

    return model

def project_simplex(n):
    fac_1 = torch.sqrt(torch.tensor([1 + 1/n])) * torch.ones(n, n)
    fac_2 = torch.pow(torch.tensor([n]), -(3/2)) * torch.ones(n, n)
    fac_3 = (torch.sqrt(torch.tensor([n + 1])) + 1) * torch.ones(n, n)

    verts = fac_1 * torch.eye(n) - fac_2 * fac_3
    extra_vert = torch.ones(1, n) * torch.pow(torch.tensor([n]), -(1/2))
    verts = torch.cat((verts, extra_vert), 0)
    return verts


def get_smiles(loader, max_nodes):
    test_smiles = []
    for mol in loader.dataset:
        # get RDKit mol
        # make dense
        mol, _ = to_dense(mol.x, mol.edge_index, mol.edge_attr, mol.batch, max_nodes)

        mol = make_molecule(mol.X.squeeze(), mol.E.squeeze(), max_nodes, None)
        smiles = Chem.MolToSmiles(mol)
        test_smiles.append(smiles)

    return test_smiles


def generate_loader(task, max_nodes, batch_size, small_data):
    if task[:3] == 'qm9':

        dataset = QM9(root='data/QM9')

        dataset = dataset.shuffle()
        dataset.shuffle()

        train_dataset = dataset[:100000]
        val_dataset = dataset[100000:120000]
        test_dataset = dataset[120000:]

        if small_data:
            train_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in train_dataset[:1000]]
            val_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in val_dataset[:1000]]
            test_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in test_dataset[:1000]]
        else:
            train_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in train_dataset]
            val_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in val_dataset]
            test_dataset = [process_graph_qm9(mol, max_nodes=max_nodes) for mol in test_dataset]

    if task == 'zinc':
        train_dataset = ZINC(root='data/ZINC', subset=False, split='train')
        val_dataset = ZINC(root='data/ZINC', subset=False, split='val')
        test_dataset = ZINC(root='data/ZINC', subset=False, split='test')

        inds = torch.tensor([0,2,1,3,0,8,5,2,1,4,1,1,1,1,8,6,7,2,1,2,8,7,7,0,7,8,0,7])

        processes_graphs = {}

        datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

        for k, v in datasets.items():
            gs = []

            for graph in tqdm(v):
                feat = graph.x
                feat = inds[feat]
                graph.x = torch.nn.functional.one_hot(feat.long(), num_classes=9).float()

                e = graph.edge_index.T.tolist()
                f = graph.edge_attr.tolist()

                edges, features = [], []
                d = {tuple(ex): ef for ex, ef in zip(e, f)}

                # edge index to list of edges

                # list of edges to set of edges
                # for i, j in product(range(graph.x.shape[0]), range(graph.x.shape[0])):
                #
                #     if (i, j) in d:
                #         edges.append([i, j])
                #
                #         fea = d[(i, j)]
                #         fea = fea + 1
                #
                #         features.append(fea)
                #
                #         print(fea)
                #
                # quit()
                edge_index = torch.tensor(edges).T
                edge_attr = torch.tensor(features)
                edge_attr = torch.nn.functional.one_hot(edge_attr.long(), num_classes=4).squeeze()

                process_graph = Data(x=graph.x.squeeze(), edge_index=graph.edge_index, edge_attr=torch.nn.functional.one_hot(graph.edge_attr.long(), num_classes=4).squeeze())

                gs.append(process_graph)

            processes_graphs[k] = gs

        train_dataset = processes_graphs['train']
        val_dataset = processes_graphs['val']
        test_dataset = processes_graphs['test']

    return train_dataset, val_dataset, test_dataset


def process_graph_qm9(mol, max_nodes):
    # get non-hydrogen nodes
    X = mol.x[:, 1:5]
    # remove all rows that where hydrogen
    non_H_nodes = ~X.eq(0).all(dim=1)
    num_non_H_nodes = torch.sum(non_H_nodes)
    X = X[non_H_nodes]
    # pad x with zeros to max_nodes
    # X = torch.cat((X, torch.zeros(size=(max_nodes - num_non_H_nodes, 4))), dim=0)
    # dict that maps each edge to its edge attribute
    non_H_indices = torch.arange(len(non_H_nodes))[non_H_nodes].tolist()
    dict = {old_i: i for i, old_i in enumerate(non_H_indices)}

    # loop over edges in index
    edge_index = mol.edge_index
    edge_attr = mol.edge_attr

    e = edge_index.T.tolist()
    f = edge_attr.tolist()

    edges, features = [], []
    d = {tuple(ex): ef for ex, ef in zip(e, f)}

    # edge index to list of edges

    # list of edges to set of edges
    for i, j in product(range(num_non_H_nodes), range(num_non_H_nodes)):


        if (i, j) in d:
            edges.append([i, j])


            fea = d[(i, j)]
            fea = np.argmax(fea) + 1

            features.append(fea)
        #
        # else:
        #     features.append(0)

    edge_index = torch.tensor(edges).T
    edge_attr = torch.tensor(features)
    edge_attr = torch.nn.functional.one_hot(edge_attr.long(), num_classes=4).squeeze()

    mask = torch.tensor([num_non_H_nodes * [True] + (max_nodes - num_non_H_nodes) * [False]]).squeeze().unsqueeze(-1)

    if X.shape[0] == 1:
        edge_attr = edge_attr.unsqueeze(0)

    return Data(x=X, edge_index=edge_index, edge_attr=edge_attr)
