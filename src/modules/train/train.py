# from builtins import breakpoint
# from builtins import breakpoint
from tqdm.auto import tqdm
import torch

def train_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device):
    decoder.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0
        for index in range(data["X"].shape[0]):
            y_grt = data["Y"][index].to(device).float()
            x = data["X"][index].to(device).float()
            G = data["G"][index].to(device).float()
            h = stdgi.embedd(x, G)
            l = data["l"][index].to(device).float()
            y_prd = decoder(x, h, l)  # 3x1x1
            batch_loss += criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss = batch_loss / data["X"].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    train_loss = epoch_loss / len(dataloader)
    return train_loss


import wandb
def train_atten_stdgi(
    stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2
):
    # wandb.watch(stdgi, criterion, log="all", log_freq=100)
    '''
    Sử dụng train Attention_STDGI model 
    '''
    epoch_loss = 0
    stdgi.train()
    for data in tqdm(dataloader): 
        for i in range(n_steps):
            optim_d.zero_grad()
            d_loss = 0
            x = data["X"].to(device).float()
            G = data["G"][:,:,:,:,0].to(device).float()  
            output = stdgi(x, x, G)
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            d_loss = criterion(output, lbl)
            d_loss.backward()
            optim_d.step()

        optim_e.zero_grad()
        x = data["X"].to(device).float()
        G = data["G"][:,:,:,:,0].to(device).float()  
        output = stdgi(x, x, G)
        lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
        lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
        lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
        e_loss = criterion(output, lbl)
        e_loss.backward()
        optim_e.step()
        epoch_loss += e_loss.detach().cpu().item()
    return epoch_loss / len(dataloader)
from src.layers.loss import linex_loss

def train_atten_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device):
    # wandb.watch(decoder, criterion, log="all", log_freq=100)
    decoder.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0
        y_grt = data["Y"].to(device).float()
        x = data["X"].to(device).float()
        G = data["G"][:,:,:,:,0].to(device).float()
        l = data["l"].to(device).float()
        cli = data['climate'].to(device).float()
        h = stdgi.embedd(x, G)
        y_prd = decoder(x, h, l,cli)  # 3x1x1
        batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    train_loss = epoch_loss / len(dataloader)
    return train_loss

def train_egcn_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device, args):
    # wandb.watch(decoder, criterion, log="all", log_freq=100)
    decoder.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0

        if args.model_type in ['gede', 'woclimate', "woaddnoise",  'wogcn']:
            y_grt = data["Y"].to(device).float()
            x = data["X"].to(device).float()
            G = data["G"].to(device).float()
            l = data["l"].to(device).float()
            cli = data['climate'].to(device).float()
        elif args.model_type == 'wornnencoder':
            y_grt = data["Y"].to(device).float()
            x = data["X"][:,-1].to(device).float()
            G = data["G"][:,-1].to(device).float()
            l = data["l"].to(device).float()
            cli = data['climate'].to(device).float()
        h = stdgi.embedd(x, G)
        y_prd = decoder(x, h, l,cli)  # 3x1x1
        batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss

    train_loss = epoch_loss / len(dataloader)
    return train_loss
    
from src.layers.loss import linex_loss

def train_egcn(
    stdgi, dataloader, optim, criterion, device, args
):
    # wandb.watch(stdgi, criterion, log="all", log_freq=100)
    epoch_loss = 0
    stdgi.train()
    for data in tqdm(dataloader):
        try:
            if args.model_type in ['gede', 'woclimate', "woaddnoise",  'wogcn']:
                x = data["X"][:,:,:,:].to(device).float()
                G = data["G"][:,:,:,:,:].to(device).float()
            elif args.model_type == 'wornnencoder':
                x = data["X"][:,-1,:,:].to(device).float()
                G = data["G"][:,-1,:,:,:].to(device).float()
            # try:
        
            output = stdgi(x, x, G)
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            # print(output)
            loss = criterion(output, lbl)
            loss.backward()
            optim.step()
        
        #     import pdb; pdb.set_trace()
        except:
            breakpoint()
        epoch_loss += loss
        
    return epoch_loss / len(dataloader)

from src.layers.loss import linex_loss


def train_end2end(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0

        for index in range(data["X"].shape[0]):
            y_grt = data["Y"][index].to(device).float()
            x = data["X"][index].to(device).float()
            G = data["G"][index][0].to(device).float()
            l = data["l"][index].to(device).float()
            cli = data['climate'][index].to(device).float()
            y_prd = model(x,G.unsqueeze(0),l,cli)  # 3x1x1
            batch_loss += criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss = batch_loss / data["X"].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.detach().cpu().item()
    train_loss = epoch_loss / len(dataloader)
    return train_loss