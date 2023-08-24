from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score , mean_absolute_percentage_error
import pandas as pd

def train_fcnn(model, dataloader, loss_fn, optimizer, device, scheduler):
    model.train() 
    total_loss = 0 
    losses = []
    for data in tqdm(dataloader):
        optimizer.zero_grad() 
        x = data['x'].to(device)
        meteo = data['meteo'].to(device) 
        y = data['y'].to(device)
        input_stat = data['input_stat'].to(device)
        target_stat = data['stat'].to(device)

        out = model(x, meteo, input_stat, target_stat)
        loss = loss_fn(out, y)
        losses.append(loss) 
        loss.backward()
        optimizer.step()
    
    total_loss = torch.tensor(losses).mean().item()
    scheduler.step(total_loss)
    return total_loss

def test_fcnn(model, dataloader, device, pm_scaler, output_dir, stat, config):
    with torch.no_grad():
        model.eval()
        gt,output = [] , []

        for data in tqdm(dataloader): 
            x = data['x'].to(device)
            meteo =data['meteo'].to(device)
            y = data['y'].to(device)
            input_stat = data['input_stat'].to(device)
            target_stat = data['stat'].to(device)
            out = model(x, meteo, input_stat, target_stat)
            gt += y.tolist() 
            output += out.tolist() 

        gt_ = np.expand_dims(np.array(gt),axis=-1)
        output_ = np.expand_dims(np.array(output), axis=-1)
        gt = pm_scaler.inverse_transform(gt_)
        output = pm_scaler.inverse_transform(output_)
        
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(np.arange(len(gt)), gt, label="groundtruth")
        ax.plot(np.arange(len(output)), output, label="predict")
        ax.legend()
        ax.set_title(f"Tram_{stat}") 
        fig.savefig(output_dir + f"stat_{stat}_bs_{config['batch_size']}.png")
        plt.close(fig)

        df_res = pd.DataFrame(data={'groundtruth': gt.squeeze(-1).tolist(), 'predict': output.squeeze(-1).tolist()})
        df_res.to_csv(output_dir + f"result_stat_{stat}_bs_{config['batch_size']}.csv")

        mse = mean_squared_error(gt, output)
        mae = mean_absolute_error(gt, output)
        mape = mean_absolute_percentage_error(gt, output)
        r2 = r2_score(gt, output)
        return {'mse': mse, 'mae': mae, 'mape':mape, 'r2':r2}
