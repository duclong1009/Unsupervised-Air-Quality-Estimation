# CUDA_VISIBLE_DEVICES=2 python train_tgcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede  --dataset "uk"  --group_name "_UK" --decoder_type default
CUDA_VISIBLE_DEVICES=0 python train_tgcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede  --dataset "uk"  --group_name "_UK" --decoder_type temporal_attention_v1
CUDA_VISIBLE_DEVICES=0 python train_tgcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede  --dataset "uk"  --group_name "_UK" --decoder_type temporal_attention_v2
CUDA_VISIBLE_DEVICES=0 python train_tgcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name " " --en_hid1 64 --en_hid2 64 --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede  --dataset "uk"  --group_name "_UK" --decoder_type temporal_attention_v3
