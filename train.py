import torch
from utils import to_dense, get_tau_sched, make_molecule, project_simplex
from flow_matching import conditional_velocity
from rdkit import Chem
from structural import add_feats

# def compute_dz_dt(alpha_0, alpha_1, t, gumbel, tau, eps=1e-10):
#     alpha_e_t = t * alpha_1 + (1 - t) * alpha_0
#     u = (torch.log(alpha_e_t + eps) + gumbel) / tau
#     du_dt = ((alpha_1 - alpha_0) / (alpha_e_t + eps)) / tau
#     z = torch.softmax(u, dim=-1)
#     diag_z = torch.diag_embed(z)
#     jacobian_z = diag_z - z.unsqueeze(-1) * z.unsqueeze(-2)
#     du_dt_reshaped = du_dt.unsqueeze(-1)
#     dz_dt = torch.matmul(jacobian_z, du_dt_reshaped)
#     return z, dz_dt.squeeze()


def train_epoch(train, model, loader, optimizer, lr_scheduler, warmup_scheduler, device, max_nodes, mu, sigma, ema, distribution, tau_sched, loss_function, accelerator):
    total_loss, x_loss, e_loss = 0, 0, 0
    if train:
        model.train()
    else:
        model.eval()

    # node_simplex = project_simplex(3)
    # edge_simplex = project_simplex(3)

    # node_simplex, edge_simplex = node_simplex.to(device), edge_simplex.to(device)

    k = 0

    for batch in loader:
        if train:
            optimizer.zero_grad()

        batch = batch.to(device)

        graph, mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch, max_nodes)
        graph = graph.mask(mask)

        diag = torch.eye(graph.X.shape[1], dtype=torch.bool).unsqueeze(0).expand(graph.X.shape[0], -1, -1)
        t = torch.rand(graph.X.shape[0], device=batch.x.device)
        tau_t = get_tau_sched(tau_sched)
        x_1, e_1 = graph.X.float(), graph.E.float()

        # for k in range(10):
        #
        #     x_1e = x_1[k]
        #     e_1e = e_1[k]
        #     x_0e = torch.randn_like(x_1e)
        #     e_0e = torch.randn_like(e_1e)
        #     e_0e = 0.5 * (e_0e + e_0e.transpose(0, 1))
        #     maske = mask[k]
        #     mols = []
        #     for t in torch.linspace(0, 1, 5):
        #         x_te = t * x_1e + (1 - t) * x_0e
        #         e_te = t * e_1e + (1 - t) * e_0e
        #
        #         mol = make_molecule(x_te, e_te, sum(maske), True)
        #         mols.append(mol)
        #         # visualize RDKit grid
        #     grid = Chem.Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(1000, 1000))
        #     grid.show()
        # quit()
        #
        #


        # x_1 = torch.einsum('bni,ij->bnj', x_1.float(), node_simplex)
        # e_1 = torch.einsum('bni,ij->bnj', e_1.float(), edge_simplex)


        # gumbel_x = -torch.log(-torch.log(torch.rand_like(x_1.float()) + 1e-10) + 1e-10)
        # gumbel_e = -torch.log(-torch.log(torch.rand_like(e_1.float()) + 1e-10) + 1e-10)
        # gumbel_e = 0.5 * (gumbel_e + gumbel_e.transpose(1, 2))
        # x_0, e_0 = torch.ones_like(x_1) / 4, torch.ones_like(e_1) / 4
        #
        # all_mols = []
        # for t in torch.linspace(0, 1, 10):
        #     x_t = torch.softmax((torch.log(t * x_1 + (1 - t) * x_0) + gumbel_x) / tau_t(t), dim=-1)
        #     e_t = torch.softmax((torch.log(t * e_1 + (1 - t) * e_0) + gumbel_e) / tau_t(t), dim=-1)
        #
        #     mols = [make_molecule(x_t[i], e_t[i], max_nodes, True) for i in range(10)]
        #     all_mols.extend(mols)
        #
        # # visualize RDKit grid
        #
        # grid = Chem.Draw.MolsToGridImage(all_mols, molsPerRow=10, subImgSize=(200, 200))
        # grid.show()
        # quit()

        t = t.unsqueeze(-1).unsqueeze(-1)
        x_t, v_x = conditional_velocity(distribution, x_1, t, tau_t, mu, sigma)
        t = t.unsqueeze(-1)
        e_t, v_e = conditional_velocity(distribution, e_1, t, tau_t, mu, sigma, edge=True)
        e_t[diag], v_e[diag] = 0, 0
        #
        # e_t_r, x_t_r = torch.argmax(e_t, dim=-1), torch.argmax(x_t, dim=-1)
        # e_t_r, x_t_r = torch.nn.functional.one_hot(e_t_r, e_t.size(-1)), torch.nn.functional.one_hot(x_t_r, x_t.size(-1))
        #
        # x_t, y_t = add_feats(x_t, e_t_r, t, mask, device)
        y_t = torch.squeeze(t).unsqueeze(-1)

        # x_t = torch.clone(x_t.detach()).to(device)
        # e_t = torch.clone(e_t.detach()).to(device)
        # v_x = torch.clone(v_x.detach()).to(device)
        # v_e = torch.clone(v_e.detach()).to(device)
        # y_t = y_t.to(device)
        # mask = mask.to(device)

        # make prediction
        pred = model(x_t.float(), e_t.float(), y_t.float(), mask)
       #  dis_# x = torch.cdist(pred.X, node_simplex, p=2)

        # pred.X = torch.nn.functional.one_hot(x_pred, x_pred.size(-1))
        # pred.E = torch.nn.functional.one_hot(e_gen, e_gen.size(-1))

        # get masks
        true_x = torch.reshape(x_1, (-1, x_1.size(-1))).to(device)
        true_e = torch.reshape(e_1, (-1, e_1.size(-1))).to(device)
        masked_pred_x = torch.reshape(pred.X, (-1, x_1.size(-1))).to(device)
        masked_pred_e = torch.reshape(pred.E, (-1, e_1.size(-1))).to(device)
        mask_x = (true_x != 0.).any(dim=-1).to(device)
        mask_e = (true_e != 0.).any(dim=-1).to(device)

        criterion = torch.nn.MSELoss() if loss_function == 'mse' else torch.nn.CrossEntropyLoss()

        if loss_function == 'mse':
            target_x = torch.reshape(v_x, (-1, x_1.size(-1))).to(device)
            target_e = torch.reshape(v_e, (-1, e_1.size(-1))).to(device)
            # target_x = torch.reshape(x_1, (-1, x_1.size(-1))).to(device)
            # target_e = torch.reshape(e_1, (-1, e_1.size(-1))).to(device)

        elif loss_function == 'kld':
            target_x = torch.reshape(x_1, (-1, x_1.size(-1))).to(device)
            target_e = torch.reshape(e_1, (-1, e_1.size(-1))).to(device)
            target_x = torch.argmax(target_x, dim=-1)
            target_e = torch.argmax(target_e, dim=-1)

        masked_pred_x, masked_pred_e = masked_pred_x[mask_x], masked_pred_e[mask_e]
        target_x, target_e = target_x[mask_x], target_e[mask_e]

        loss_X = criterion(masked_pred_x.float(), target_x)
        loss_E = criterion(masked_pred_e.float(), target_e)

        loss = loss_X + 5 * loss_E

        if train:
            # accelerator.backward(loss)
            loss.backward()

            max_grad_norm = 1.0
            # accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()

            # ema = ema.to(batch.x.device)
            if ema:
                ema.update()

        total_loss += loss.item() * (graph.X.shape[0])
        e_loss += loss_E.item() * (graph.X.shape[0])
        x_loss += loss_X.item() * (graph.X.shape[0])


    train_loss = total_loss / len(loader.dataset)
    train_x_loss = x_loss / len(loader.dataset)
    train_e_loss = e_loss / len(loader.dataset)

    return model, train_loss, train_x_loss, train_e_loss
