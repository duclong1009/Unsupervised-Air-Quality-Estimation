# Overview
This is the official code of our proposal method GEDE with PyTorch implementation

# Abstract
The rapid progress of industrial development, urbanization, and traffic have caused air quality degradation that negatively affects human health and environmental sustainability, especially in developed countries. However, due to the limited number of sensors available, the air quality index at many locations is not monitored. Therefore, many research, including statistical and machine learning approaches, have been proposed to tackle the problem of estimating air quality value at an arbitrary location. Most of the existing research perform interpolation process based on traditional techniques that leverage distance information. In this work, we propose a novel deep-learning-based model for air quality value estimation. This approach follows the encoder-decoder paradigm, with the encoder and decoder trained separately using different training mechanisms. In the encoder component, we proposed a new self-supervised graph representation learning approach for spatio-temporal data. For the decoder component, we designed a deep interpolation layer that employs two attention mechanisms and a fully-connected layer using air quality data at known stations, distance information, and meteorology information at the target point to predict air quality at arbitrary locations. The experimental results show that our proposed model increases the estimation accuracy from 4.93\% to 34.88\% in MAE metrics compared to the state-of-the-art approaches

![plot](image/gede.png)
![plot](image/accuracy.png)
##  Install package
```
pip install -r requirements.txt
```
## Data
Download data from [link](https://drive.google.com/file/d/1JoNoc_EDzV2EOH1vFhYcgMjOQgu-MSMz/view?usp=sharing)
Download this file, then decompress this file and locate it at the path like that /path-to-repo/data

This file contains data from Beijing(China) and the UK.
## Training 
To train the model from scratch with the default setting
```
python train.py -stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name "name_of_running_session" --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede --dataset beijing --group_name "name_of_wandb_group" --decoder_type localglobal_attention --features "PM2.5,CO,SO2,2m_temperature,evaporation"
```


## Options
* ```--train_station``` List index of training station .
* ```--test_station``` List index of testing station.
* ```--output_dim``` Default:1.Output dimension of decoder.
* ```--sequence_length``` Default:12. Number historical timestep used for each sample.
* ```--patience``` Count parameter for early stopping.
* ```--climate_features``` List of meteorology features is used in Decoder
#### Stdgi hyperparameter
* ```--checkpoint_stdgi``` Checkpoint path for saving the stdgi's model weight
* ```--output_stdgi``` Dimension of output of stdgi 
* ```--lr_stdgi``` Learning rate.
* ```--num_epochs_stdgi``` Number epochs to train stdgi
* ```--checkpoint_stdgi``` Saved path for stdgi weight.
* ```--en_hid1```, ```--en_hid2``` Dimension of GCN in Encoder 
* ```--dis_hid```Dimension of FC layer in Encoder
* ```--act_fn``` Activation function is used.
* ```--delta_stdgi``` delta parameter for EarlyStopping
*

#### Decoder hyperparameter
* ```lr_decoder``` Learning rate.
* ```num_epochs_decoder``` Number epochs for training decoder.
* ```checkpoint_decoder``` Saved path for decoder's weight
* ```delta_decoder``` Delta parameter of EarlyStopping.
* ```cnn_hid_dim```Dimension of CNN layers 
* ```fc_hid_dim``` Dimension of FC layer in Decoder
 
