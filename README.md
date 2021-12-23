# Overview
Project for unsupervised prediction in spatial


##  Install package


## Run model
```
conda activate longvaipe
```
```
python train.py
```

## Options
* ```--train_station``` Trạm được dùng trong quá trình train stdgi và decoder.
* ```--test_station``` Trạm được dùng trong quá trình test.
* ```--input_dim``` Default:1. Số features của input(stdgi).
* ```--output_dim``` Default:1. Chiều đầu ra của output(decoder).
* ```--sequence_length``` Default:12. Số time step của 1 sample.
#### Stdgi hyperparameter
* ```lr_stdgi``` Learning rate.
* ```num_epochs_stdgi``` Số epoch để train stdgi.
* ```checkpoint_stdgi``` địa chỉ lưu model stdgi.
* ```en_hid1```, ```en_hid2``` Số chiều của của các lớp GCN trong Encoder.
* ```dis_hid``` Số chiều của Fc trong Discriminator.
* ```act_fn``` Hàm kích hoạt của lớp GCN.
* ```delta_stdgi``` Tham số sử dụng trong EarlyStopping. Model cải thiện khi loss+delta < best_loss.
*

#### Decoder hyperparameter
* ```lr_decoder``` Learning rate.
* ```num_epochs_decoder``` Số lượng epochs train decoder.
* ```checkpoint_decoder``` Địa chỉ lưu model decoder.
* ```delta_decoder``` Tham số sử dụng trong EarlyStopping.
* ```cnn_hid_dim``` Số channel của lớp CNN 
* ```fc_hid_dim``` Số chiều lớp FC trong Decoder
* ```n_layers_rnn``` Số layer của lớp RNN
* ```rnn_type``` Kiểu của lớp RNN có thể là RNN,GRU,LSTM
 
## Model 
* train_attention_stdgi : Train Attention STDGI sử dụng hàm train_atten_stdgi, train_atten_decoder_fn. Sử dụng train 2 disc , 1 lần encoder
* train_trick_gan : Sử dụng cách train 2 disc, 1 lần encoder 
* train.py: Sử dụng cách train bình thường