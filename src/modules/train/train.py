from builtins import breakpoint
from tqdm.auto import tqdm
import torch


def train_stdgi_fn(stdgi, dataloader, optimizer, criterion, device, interpolate=False):
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0
        for index in range(data["X"].shape[0]):
            x = data["X"][index].to(device).float()
            G = data["G"][index].to(device).float()
            l = data["l"][index].to(device).float()
            lbl_1 = torch.ones(x.shape[0], x.shape[1], 1)
            lbl_2 = torch.zeros(x.shape[0], x.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), 2).to(device)
            if not interpolate:
                output = stdgi(x, x, G)
            else:
                output = stdgi(x,x, G,l)
                
            batch_loss += criterion(output, lbl)
        batch_loss = batch_loss / data["X"].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    return epoch_loss / len(dataloader)


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


def train_stdgi_with_trick_fn(
    stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2
):
    stdgi.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        loss = 0
        for i in range(n_steps):
            optim_d.zero_grad()
            d_loss = 0
            for index in range(data["X"].shape[0]):
                x = data["X"][index].to(device).float()
                G = data["G"][index].to(device).float()
                lbl_1 = torch.ones(x.shape[0], x.shape[1], 1)
                lbl_2 = torch.zeros(x.shape[0],x.shape[1], 1)
                lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
                output = stdgi(x, x, G)
                d_loss += criterion(output, lbl)
            d_loss = d_loss / data["X"].shape[0]
            d_loss.backward()
            optim_d.step()
        # khong su dung batch
        optim_e.zero_grad()
        for index in range(data["X"].shape[0]):
            x = data["X"][index].to(device).float()
            G = data["G"][index].to(device).float()
            lbl_1 = torch.ones(x.shape[0], x.shape[1], 1)
            lbl_2 = torch.zeros(x.shape[0], x.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            output = stdgi(x, x, G)
            d_loss += criterion(output, lbl)
            # import pdb; pdb.set_trace()
            loss += criterion(output, lbl)
        loss = loss / data["X"].shape[0]
        loss.backward()
        optim_e.step()
        epoch_loss += loss
    return epoch_loss / len(dataloader)

import wandb
def train_atten_stdgi(
    stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2, interpolate=False
):
    # wandb.watch(stdgi, criterion, log="all", log_freq=100)
    '''
    Sử dụng train Attention_STDGI model 
    '''
    epoch_loss = 0
    stdgi.train()
    for data in tqdm(dataloader):
        e_loss = 0  
        for i in range(n_steps):
            optim_d.zero_grad()
            d_loss = 0
            for index in range(data["X"].shape[0]):
                x = data["X"][index].to(device).float()
                G = data["G"][index][0].to(device).float()  
                l = data["l"][index].to(device).float()
                output = stdgi(x, x, G.unsqueeze(0))
                lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
                lbl_2 = torch.zeros(output.shape[0],output.shape[1], 1)
                lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
                d_loss += criterion(output, lbl)
            d_loss = d_loss / data["X"].shape[0]
            d_loss.backward()
            optim_d.step()
        # khong su dung batch
        optim_e.zero_grad()
        for index in range(data["X"].shape[0]):
            x = data["X"][index].to(device).float()
            G = data["G"][index][0].to(device).float()
            l = data["l"][index].to(device).float()
            output = stdgi(x, x, G.unsqueeze(0))
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            e_loss += criterion(output, lbl)
        e_loss = e_loss / data["X"].shape[0]
        e_loss.backward()
        optim_e.step()
        epoch_loss += e_loss.detach().cpu().item()
    return epoch_loss / len(dataloader)
from src.layers.loss import linex_loss

def train_atten_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device, interpolate=False):
    decoder.train()
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
            h = stdgi.embedd(x, G.unsqueeze(0))
            y_prd = decoder(x[-1].unsqueeze(0), h, l,cli)  # 3x1x1
            batch_loss += criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss = batch_loss / data["X"].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    train_loss = epoch_loss / len(dataloader)
    return train_loss

def train_egcn_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device, interpolate=False):
    # wandb.watch(decoder, criterion, log="all", log_freq=100)
    decoder.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0

        for index in range(data["X"].shape[0]):
            y_grt = data["Y"][index].to(device).float()
            x = data["X"][index][-1].unsqueeze(0).to(device).float()
            G = data["G"][index][-1].unsqueeze(0).to(device).float()
            l = data["l"][index].to(device).float()
            cli = data['climate'][index].to(device).float()
            h = stdgi.embedd(x, G)
            y_prd = decoder(x[-1].unsqueeze(0), h, l,cli)  # 3x1x1
            batch_loss += criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss = batch_loss / data["X"].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    train_loss = epoch_loss / len(dataloader)
    return train_loss
    
# def train_atten_stdgi(
#     stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2, interpolate=False
# ):
#     # wandb.watch(stdgi, criterion, log="all", log_freq=100)
#     '''
#     Sử dụng train Attention_STDGI model 
#     '''
#     # stdgi = STDGI(12,60, gconv=gconv).to(device)
#     epoch_loss = 0
#     stdgi.train()
#     for data in tqdm(dataloader):
#         e_loss = 0  
#         for i in range(n_steps):
#             optim_d.zero_grad()
#             loss = 0
#             list_output= []
#             list_lbl = []
#             for index in range(data["X"].shape[0]):
#                 x = data["X"][index].to(device).float()
#                 G = data["G"][index][0].to(device).float()  
#                 l = data["l"][index].to(device).float()
#                 output = stdgi(x, x, G.unsqueeze(0))
#                 lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
#                 lbl_2 = torch.zeros(output.shape[0],output.shape[1], 1)
#                 lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
#                 list_output.append(output)
#                 list_lbl.append(lbl)
#             output_ = torch.concat(list_output, dim=0)
#             lbl_ = torch.concat(list_lbl, dim=0)
#             loss += criterion(output_, lbl_)
#             loss.backward()
#             optim_d.step()
#             del list_lbl,list_output
#         # khong su dung batch
#         optim_e.zero_grad()
#         list_output= []
#         list_lbl = []
#         for index in range(data["X"].shape[0]):
#             x = data["X"][index].to(device).float()
#             G = data["G"][index][0].to(device).float()
#             l = data["l"][index].to(device).float()
#             output = stdgi(x, x, G.unsqueeze(0))
#             lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
#             lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
#             lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
#             # import pdb; pdb.set_trace()
#             list_output.append(output)
#             list_lbl.append(lbl)
#         output_ = torch.concat(list_output, dim=0)
#         lbl_ = torch.concat(list_lbl, dim=0)
#         del list_output, list_lbl
#         e_loss = criterion(output_, lbl_)
#         e_loss.backward()
#         optim_e.step()
#         epoch_loss += e_loss
#     return epoch_loss / len(dataloader)
from src.layers.loss import linex_loss



def train_egcn(
    stdgi, dataloader, optim, criterion, device
):
    # wandb.watch(stdgi, criterion, log="all", log_freq=100)
    epoch_loss = 0
    stdgi.train()
    for data in tqdm(dataloader):
        # breakpoint()
        x = data["X"][:,-1,:,:].to(device).float()
        G = data["G"][:,-1,:,:,:].to(device).float()
        
        lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
        lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
        lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
        try:
            output = stdgi(x, x, G)
            loss = criterion(output, lbl)
            loss.backward()
        except:
            breakpoint()
        optim.step()
        epoch_loss += loss
        
    return epoch_loss / len(dataloader)

from src.layers.loss import linex_loss

# def train_atten_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device, interpolate=False):
#     # wandb.watch(decoder, criterion, log="all", log_freq=100)
#     decoder.train()
#     epoch_loss = 0
#     for data in tqdm(dataloader):
#         optimizer.zero_grad()
#         batch_loss = 0

#         for index in range(data["X"].shape[0]):
#             y_grt = data["Y"][index].to(device).float()
#             x = data["X"][index].to(device).float()
#             G = data["G"][index][0].to(device).float()
#             l = data["l"][index].to(device).float()
#             cli = data['climate'][index].to(device).float()
#             if not interpolate:
#                 h = stdgi.embedd(x, G.unsqueeze(0))
#                 y_prd = decoder(x[-1].unsqueeze(0), h, l,cli)  # 3x1x1
#             else:
#                 h, enc_hidd = stdgi.embedd(x, G.unsqueeze(0), l)
#                 # import pdb; pdb.set_trace()
#                 y_prd = decoder(x[-1].unsqueeze(0), h, l)
#             batch_loss += criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
#         batch_loss = batch_loss / data["X"].shape[0]
#         batch_loss.backward()
#         optimizer.step()
#         epoch_loss += batch_loss.item()
#     train_loss = epoch_loss / len(dataloader)
#     return train_loss


def train_end2end(model, dataloader, criterion, optimizer, device, interpolate=False):
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