# gede
CUDA_VISIBLE_DEVICES=0 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name gede_uk --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_gede_uk --checkpoint_stdgi stdgi_gede_uk --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede --dataset uk
# wogcn
CUDA_VISIBLE_DEVICES=0 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name gede_uk --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_wogcn_uk --checkpoint_stdgi stdgi_wogcn_uk --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type wogcn --dataset uk
# wornnencoder
CUDA_VISIBLE_DEVICES=0 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name gede_uk --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_wornnencoder_uk --checkpoint_stdgi stdgi_wornnencoder_uk --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type wornnencoder --dataset uk
# remove noise 
CUDA_VISIBLE_DEVICES=0 python train_egcn.py --stdgi_noise_min 1 --stdgi_noise_max 1 --name egcn --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_remove_noise_uk --checkpoint_stdgi stdgi_remove_noise_uk --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede --dataset uk
# use_wind
CUDA_VISIBLE_DEVICES=1 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name egcn --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_wowind_uk --checkpoint_stdgi stdgi_wowind_uk --num_epochs_stdgi 1 --num_epochs_decoder 1 --model_type gede --use_wind --dataset uk 
# woclimate
CUDA_VISIBLE_DEVICES=0 python train_egcn.py --stdgi_noise_min 0.4 --stdgi_noise_max 0.8 --name egcn --en_hid1 64 --en_hid2 64 --checkpoint_decoder decoder_woclimate_uk --checkpoint_stdgi stdgi_woclimate_uk --num_epochs_stdgi 30 --num_epochs_decoder 30 --model_type gede --wo_climate --dataset uk