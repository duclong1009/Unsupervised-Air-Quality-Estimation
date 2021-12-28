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
    # stdgi = STDGI(12,60, gconv=gconv).to(device)
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


def train_atten_stdgi(
    stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2, interpolate=False
):
    '''
    Sử dụng train Attention_STDGI model 
    '''
    # stdgi = STDGI(12,60, gconv=gconv).to(device)
    epoch_loss = 0
    for data in tqdm(dataloader):
        loss = 0
        for i in range(n_steps):
            optim_d.zero_grad()
            d_loss = 0
            for index in range(data["X"].shape[0]):
                x = data["X"][index].to(device).float()
                G = data["G"][index][0].to(device).float()  
                l = data["l"][index].to(device).float()
                if not interpolate:
                    output = stdgi(x, x, G.unsqueeze(0))
                else:
                    # import pdb; pdb.set_trace()
                    output, _ = stdgi(x, x, G.unsqueeze(0), l)
                    # print(output.shape)
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
            if not interpolate:
                output = stdgi(x, x, G.unsqueeze(0))
            else:
                output, _ = stdgi(x, x, G.unsqueeze(0), l)
            lbl_1 = torch.ones(output.shape[0], output.shape[1], 1)
            lbl_2 = torch.zeros(output.shape[0], output.shape[1], 1)
            lbl = torch.cat((lbl_1, lbl_2), -1).to(device)
            d_loss += criterion(output, lbl)
            # import pdb; pdb.set_trace()
            loss += criterion(output, lbl)
        loss = loss / data["X"].shape[0]
        loss.backward()
        optim_e.step()
        epoch_loss += loss
    return epoch_loss / len(dataloader)

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
            if not interpolate:
                h = stdgi.embedd(x, G.unsqueeze(0))
                y_prd = decoder(x[-1].unsqueeze(0), h, l)  # 3x1x1
            else:
                h, enc_hidd = stdgi.embedd(x, G.unsqueeze(0), l)
                y_prd = decoder(x[-1].unsqueeze(0), h, enc_hidd, l)
            batch_loss += criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
        batch_loss = batch_loss / data["X"].shape[0]
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss.item()
    train_loss = epoch_loss / len(dataloader)
    return train_loss
