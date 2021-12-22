from tqdm.auto import tqdm
import torch


def train_stdgi_fn(stdgi, dataloader, optimizer, criterion, device):
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0
        for index in range(data["X"].shape[0]):
            x = data["X"][index].to(device).float()
            G = data["G"][index].to(device).float()
            lbl_1 = torch.ones(x.shape[0], 27, 1)
            lbl_2 = torch.zeros(x.shape[0], 27, 1)
            lbl = torch.cat((lbl_1, lbl_2), 2).to(device)
            output = stdgi(x, x, G)
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
    stdgi, dataloader, optim_e, optim_d, criterion, device, n_steps=2, gconv="gcn"
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
                lbl_1 = torch.ones(x.shape[0], 27, 1)
                lbl_2 = torch.zeros(x.shape[0], 27, 1)
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
            lbl_1 = torch.ones(x.shape[0], 27, 1)
            lbl_2 = torch.zeros(x.shape[0], 27, 1)
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
