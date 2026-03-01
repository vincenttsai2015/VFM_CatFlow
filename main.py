import copy
import torch
import wandb
import pandas as pd
import torch.nn as nn
import argparse
import pytorch_warmup as warmup
from rdkit.Chem import Draw
from tqdm import tqdm
from train import train_graph_epoch, train_image_epoch
from utils import count_parameters, get_loaders, make_molecule, get_smiles, get_GT_model, get_CNN_model
from torch_ema import ExponentialMovingAverage
from flow_matching import generate_graphs, generate_mnist_images_odeint, eval_and_log_molecules, eval_and_log_images
from rdkit import Chem
from utils import to_dense
import numpy as np
from accelerate import Accelerator
from rdkit import RDLogger


def main(args):
    RDLogger.DisableLog('rdApp.*')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    run = wandb.init(project=f'Cute Cats - {args.task} - {args.info}', config=vars(args), name=f'{args.num_layers}-{args.loss_function}-{args.data_size}') if args.log else None

    name = run.name if args.log else 'anom'
    train_loader, val_loader, test_loader, node_feats, edge_feats, max_nodes = get_loaders(args)

    counter = torch.zeros(max_nodes + 1)

    if args.task in ['qm9_wo_H', 'qm9', 'zinc']:
        graph_task = True
    else:
        graph_task = False
    
    if graph_task:
        for mol in tqdm(train_loader.dataset, desc='Counting nodes'):
            # print(f'mol: {mol}')
            num = mol.x.shape[0]
            counter[num] += 1
    else:
        for batch in tqdm(train_loader, desc='Counting image pixels'):
            num = batch[0].shape[1] * batch[0].shape[2]
            counter[num] += 1

    counter = counter.to(device)

    if graph_task:
        test_smiles = []
        train_smiles = get_smiles(train_loader, max_nodes)
        test_smiles = get_smiles(test_loader, max_nodes)
        smiles = (train_smiles, test_smiles)

        model = get_GT_model(args, node_feats, edge_feats)
        print(f'Number of parameters: {count_parameters(model)}')
        model.to(device)
    else:
        model = get_CNN_model()
        print(f'Number of parameters: {count_parameters(model)}')
        model.to(device)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999) if args.ema > 0 else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-12)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=2000) if args.scheduler else None
    best_model, best_loss = copy.deepcopy(model), float('inf')
    pbar = tqdm(range(args.epochs), desc='Epoch', position=0)

    if args.small_model == 1:
        MIXED_PRECISION_TYPE = 'fp16'
    else:
        MIXED_PRECISION_TYPE = 'bf16'  # of bf16
    # accelerator = Accelerator(mixed_precision=MIXED_PRECISION_TYPE, cpu=True)
    # model, optimizer, train_loader, val_loader, test_loader, ema, warmup_scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader, ema, warmup_scheduler)
    # ema.to(train_loader.dataset[0].x.device)
    accelerator = None

    for epoch in pbar:
        if graph_task:
            model, train_loss, train_loss_x, train_loss_e = train_graph_epoch(train=True, model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                                        warmup_scheduler=warmup_scheduler, device=device, max_nodes=max_nodes, mu=args.mu, sigma=args.sigma, ema=ema, 
                                                                        distribution=args.distribution, tau_sched=args.tau_sched, loss_function=args.loss_function, accelerator=accelerator)

            if args.ema > 0:
                with ema.average_parameters():
                    model, val_loss, val_loss_x, val_loss_e = train_graph_epoch(train=False, model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                                        warmup_scheduler=warmup_scheduler, device=device, max_nodes=max_nodes, mu=args.mu,  sigma=args.sigma, ema=ema, 
                                                                        distribution=args.distribution, tau_sched=args.tau_sched, loss_function=args.loss_function, accelerator=accelerator)
            
            else:
                model, val_loss, val_loss_x, val_loss_e = train_graph_epoch(train=False, model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                                        warmup_scheduler=warmup_scheduler, device=device, max_nodes=max_nodes, mu=args.mu,  sigma=args.sigma, ema=ema, 
                                                                        distribution=args.distribution, tau_sched=args.tau_sched, loss_function=args.loss_function, accelerator=accelerator)
            
            if args.log:
                wandb.log({
                    'Train Loss': train_loss,
                    'Train X Loss': train_loss_x,
                    'Train E Loss': train_loss_e,
                    'Val Loss': val_loss,
                    'Val X Loss': val_loss_x,
                    'Val E Loss': val_loss_e,
                    'Learning Rate': optimizer.param_groups[0]['lr']
                })
        else:
            model, train_loss = train_image_epoch(train=True, model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                                        warmup_scheduler=warmup_scheduler, device=device, max_nodes=max_nodes, mu=args.mu, sigma=args.sigma, ema=ema, 
                                                                        distribution=args.distribution, tau_sched=args.tau_sched, loss_function=args.loss_function, accelerator=accelerator)

            if args.ema > 0:
                with ema.average_parameters():
                    model, val_loss = train_image_epoch(train=False, model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                                        warmup_scheduler=warmup_scheduler, device=device, max_nodes=max_nodes, mu=args.mu,  sigma=args.sigma, ema=ema, 
                                                                        distribution=args.distribution, tau_sched=args.tau_sched, loss_function=args.loss_function, accelerator=accelerator)
            
            else:
                model, val_loss = train_image_epoch(train=False, model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                                        warmup_scheduler=warmup_scheduler, device=device, max_nodes=max_nodes, mu=args.mu,  sigma=args.sigma, ema=ema, 
                                                                        distribution=args.distribution, tau_sched=args.tau_sched, loss_function=args.loss_function, accelerator=accelerator)
            
            if args.log:
                wandb.log({
                    'Train Loss': train_loss, 'Val Loss': val_loss,
                    'Learning Rate': optimizer.param_groups[0]['lr']
                })

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)

        lr_scheduler.step()

        if (epoch+1) % 100 == 0:
        # if (epoch+1) in [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000]:
            if graph_task:
                generated_mols = generate_graphs(best_model, args.generate_size, node_feats, edge_feats, max_nodes, device, name, args.mu, args.distribution, args.tau_sched, args.loss_function, counter, args.small_model)
                val_mols = eval_and_log_molecules(generated_mols, args.log, smiles, device)

                for k in val_mols.keys():
                    try:
                        img = Draw.MolsToGridImage(generated_mols[k][:100], molsPerRow=10)
                        time = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        # log to wandb
                        if args.log:
                            wandb.log({f'Generated Molecules {k}': wandb.Image(img)})

                        img.save(f'images/{name}_epoch_all_{time}_{k}.png')
                        img.show()
                    except:
                        continue

                    if len(val_mols[k]) > 0:
                        img = Draw.MolsToGridImage(val_mols[k][:100], molsPerRow=10)
                        if args.log:
                            wandb.log({f'Valid Molecules {k}': wandb.Image(img)})
                        img.save(f'images/{name}_epoch_val_{time}_{k}.png')
                        img.show()
            else:
                generated_images = generate_mnist_images_odeint(best_model, args.generate_size, device)
                eval_and_log_images(generated_images, best_model, args.log, test_loader, device)

        pbar.set_postfix({'Train loss': train_loss, 'Val loss': val_loss})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-4)

    # add bool argument default false
    parser.add_argument('--scheduler', action='store_true', default=False)
    # parser.add_argument('--small_model', action='store_true', default=False)
    # parser.add_argument('--small_data', action='store_true', default=False)
    parser.add_argument('--small_model', type=int, default=0)
    parser.add_argument('--small_data', type=int, default=0)
    parser.add_argument('--data_size', type=float, default=0.01)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--mu', type=float, default=8)
    parser.add_argument('--sigma', type=float, default=0)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--task', type=str, default='qm9_wo_H')
    parser.add_argument('--generate_size', type=int, default=1000)

    parser.add_argument('--ema', type=float, default=0.999)
    parser.add_argument('--distribution', type=str, default='normal')
    parser.add_argument('--tau_sched', type=str, default='constant')
    parser.add_argument('--loss_function', type=str, default='kld')
    parser.add_argument('--info', type=str, default='')

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parsed_args.small_model = parsed_args.small_model == 1
    parsed_args.small_data = parsed_args.small_data == 1

    main(parsed_args)
