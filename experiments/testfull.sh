# gede
# CUDA_VISIBLE_DEVICES=1 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name gede_beijing --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_gede_beijing --checkpoint_stdgi stdgi_gede_beijing --num_epochs_stdgi 1 --num_epochs_decoder 1 --model_type gede --dataset beijing --log_wandb
# wogcn
# CUDA_VISIBLE_DEVICES=1 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name gede_beijing --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_wogcn_beijing --checkpoint_stdgi stdgi_wogcn_beijing --num_epochs_stdgi 1 --num_epochs_decoder 1 --model_type wogcn --dataset beijing --log_wandb
# wornnencoder
CUDA_VISIBLE_DEVICES=1 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name gede_beijing --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_wornnencoder_beijing --checkpoint_stdgi stdgi_wornnencoder_beijing --num_epochs_stdgi 1 --num_epochs_decoder 1 --model_type wornnencoder --dataset beijing --log_wandb
# remove noise 
# CUDA_VISIBLE_DEVICES=1 python train_egcn.py --stdgi_noise_min 1 --stdgi_noise_max 1 --name egcn --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_remove_noise_beijing --checkpoint_stdgi stdgi_remove_noise_beijing --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede --dataset beijing --log_wandb
# wowind
# CUDA_VISIBLE_DEVICES=1 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name egcn --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_wowind_beijing --checkpoint_stdgi stdgi_wowind_beijing --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede --use_wind --dataset beijing --log_wandb
# woclimate
# CUDA_VISIBLE_DEVICES=1 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name egcn --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_woclimate_beijing --checkpoint_stdgi stdgi_woclimate_beijing --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede --wo_climate --dataset beijing --log_wandb