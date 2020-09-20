CUDA_VISIBLE_DEVICES=1,2,3 python yahoo_with_mixtext/train_tmix_inter_lada.py --knn 3 --mu 0.5
CUDA_VISIBLE_DEVICES=1,2,3 python yahoo_with_mixtext/train_tmix_inter_lada.py --knn 5 --mu 0.5
CUDA_VISIBLE_DEVICES=1,2,3 python yahoo_with_mixtext/train_tmix_inter_lada.py --knn 3 --mu 0.75
CUDA_VISIBLE_DEVICES=1,2,3 python yahoo_with_mixtext/train_tmix_inter_lada.py --knn 5 --mu 0.75
CUDA_VISIBLE_DEVICES=1,2,3 python yahoo_with_mixtext/train_tmix_inter_lada.py --knn 5 --mu 0.5 --mix-layers 7 --mix-layers 9 --mix-layers 12