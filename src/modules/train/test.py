import torch
from tqdm import tqdm

def test_atten_decoder_fn(stdgi, decoder, dataloader, criterion, device, interpolate=False):
    decoder.eval()
    stdgi.eval()
    epoch_loss = 0
    list_prd = []
    list_grt = []
    with torch.no_grad():
        for data in tqdm(dataloader):
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
                list_prd.append(y_prd)
                list_grt.append(y_grt)
            batch_loss = batch_loss / data["X"].shape[0]
            epoch_loss += batch_loss.item()
        train_loss = epoch_loss / len(dataloader)
    return train_loss, list_prd, list_grt
