from tqdm.auto import tqdm
import torch


def train_stdgi_fn(stdgi, dataloader, optimizer, criterion, device, gconv="gcn"):
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        loss = 0
        x = data.x.to(device).float()
        edge_idx = data.edge_index.to(device)
        edge_weight = data.edge_attr.to(device)

        lbl_1 = torch.ones(x.shape[0], 1)
        lbl_2 = torch.zeros(x.shape[0], 1)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

        output = stdgi(x, x, edge_idx=edge_idx, edge_attr=edge_weight)
        loss = criterion(output, lbl)
        loss.backward()
        optimizer.step()
        epoch_loss += loss
    return epoch_loss / len(dataloader)


def train_decoder_fn(stdgi, decoder, dataloader, criterion, optimizer, device):
    decoder.train()
    epoch_loss = 0
    for data in tqdm(dataloader):
        optimizer.zero_grad()
        batch_loss = 0
        x = data.x.to(device).float()
        y = data.y.to(device).float()
        edge_idx = data.edge_index.to(device)
        edge_weight = data.edge_attr.to(device)
        l = data.pos
        h = stdgi.embedd(x, x, edge_idx=edge_idx, edge_attr=edge_weight)
        x = x.reshape(-1, 27, 12)
        h = torch.reshape(h, (-1, 27, 60))
        l = torch.tensor(l).to(device).float()
        l = torch.unsqueeze(l, -1)
        y_pred = decoder(x, h, l)
        loss = criterion(torch.squeeze(y_pred), y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)
