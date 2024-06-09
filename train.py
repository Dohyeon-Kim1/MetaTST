import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_kobart(model, args, dataset, optim, lr_scheduler=None):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    model.train()
    for epoch in range(args.epoch):
        epoch_loss = []
        pbar = tqdm(dataloader)
        for enc_input, dec_input, style in pbar:
            pred = model(enc_input, dec_input, style)
            label = model.get_label(dec_input)
            loss = model.get_loss(pred[:,:-1,:], label)
            epoch_loss.append(loss)

            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.set_postfix(loss=loss.item())

        if lr_scheduler:
            lr_scheduler.step()

        epoch_loss = torch.stack(epoch_loss).mean().item()
        print(f"Epoch {epoch+1}/{args.epoch}  train loss: {epoch_loss}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint/{args.model_name}{epoch+1}.pth")

        print(f"style: {style[0]}")
        print(f"answer: {dec_input[0]}")
        print(f"generate: {model.generate([enc_input[0]], [style[0]])}")


def train_meta_kobart(model, args, dataset, optim, lr_scheduler=None):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(args.epoch):
        epoch_loss = []
        pbar = tqdm(range(100))
        for _ in pbar:
            iter_loss = []
            for _ in range(args.meta_batch_size):
                spt, qry, _ = dataset[0]
                fast_weights = model.inner_loop(spt[0], spt[1])
                pred = model.functional_forward(qry[0], qry[1], fast_weights)
                label = model.get_label(qry[1])
                loss = model.get_loss(pred[:,:-1,:], label)
                iter_loss.append(loss)
            iter_loss = torch.stack(iter_loss).mean()
            epoch_loss.append(iter_loss.detach())

            optim.zero_grad()
            iter_loss.backward()
            optim.step()

            pbar.set_postfix(loss=iter_loss.item())

        if lr_scheduler:
            lr_scheduler.step()

        epoch_loss = torch.stack(epoch_loss).mean().item()
        print(f"Epoch {epoch+1}/{args.epoch}  train loss: {epoch_loss}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint/{args.model_name}{epoch+1}.pth")
        
        print(f"answer: {qry[1][0]}")
        print(f"generate: {model.generate([qry[0][0]], fast_weights)}")