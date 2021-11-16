import torch 
import torch.nn as nn
import tqdm 

def train_stdgi_fn(stdgi,dataloader,optimizer,criterion,device):
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0 
        # print(data['Y'].shape)
        for index in range(data['X'].shape[0]):
          x = data['X'][index].to(device).float()
          # G = data['G'][index].to(device).float()
          knn_G = data['knn_G'][index].to(device).float()
          lbl_1 = torch.ones(x.shape[0], 27,1)
          lbl_2 = torch.zeros(x.shape[0], 27,1)
          lbl = torch.cat((lbl_1, lbl_2), 2).to(device)
          # print(f"{x.shape} ; {y.shape}  {G.shape}")
          output = stdgi(x,x,knn_G)
          batch_loss += criterion(output,lbl)
        batch_loss = batch_loss/data['X'].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    return epoch_loss/len(dataloader)

def train_decoder_fn(stdgi,decoder,dataloader,criterion,optimizer,device):
    decoder.train()
    epoch_loss =0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0
        for index in range(data['X'].shape[0]):
            y_grt = data['Y'][index].to(device).float()
            x = data['X'][index].to(device).float()
            G = data['G'][index].to(device).float()
            h = stdgi.embedd(x,G)
            l = data['l'][index].to(device).float()
            y_prd = decoder(x,h,l) #3x1x1
            batch_loss += criterion(torch.squeeze(y_prd),torch.squeeze(y_grt))
        batch_loss = batch_loss/data['X'].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    train_loss = epoch_loss/len(dataloader)
    return train_loss