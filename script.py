import os

# # dataset with noise points
os.system('CUDA_VISIBLE_DEVICES=1 python main.py --mode part --name outlier_bunny --d_iter 30 --syn_outlier 600 --source dataset/bunny-x.txt --train_type m --mass 500 --beta 2 --Lambda 0.01 --sigma 0.1 --epoch 1500 --refine 1 --Refine_step 500 --vis 0 --save_pc 1')

# # partially overlapped dataset (cropping)
os.system('CUDA_VISIBLE_DEVICES=1 python main.py --mode partial --name partial_bunny_m --d_iter 30  --source dataset/bunny-x.txt --train_type m --mass 400 --beta 2 --Lambda 0.01 --sigma 0.1 --epoch 2000 --syn_subsample 1000 --normalize 0 --syn_Lambda 50 --syn_partial 0.7 --refine 1 --Refine_step 500  --vis 0 --save_pc 1')
os.system('CUDA_VISIBLE_DEVICES=1 python main.py --mode partial --name partial_bunny_h --d_iter 30  --source dataset/bunny-x.txt --train_type h --h 0.05 --beta 2 --Lambda 0.01 --sigma 0.1 --epoch 2000 --syn_subsample 1000 --normalize 0 --syn_Lambda 50 --syn_partial 0.7 --refine 1 --Refine_step 500 --vis 0 --save_pc 1')

# # 2D point sets
os.system('CUDA_VISIBLE_DEVICES=1 python main.py --mode part --name outlier_fish --d_iter 30 --test_mode 0 --source dataset/fish_X_nohead.txt --reference dataset/fish_y.txt --train_type m --mass 500 --m 50 --beta 2 --Lambda 0.01 --sigma 0.1 --epoch 10000 --refine 0 --normalize 0 --disp 20 --vis 0 --save_pc 1')



# Large armadillo dataset
os.system('CUDA_VISIBLE_DEVICES=1 python main.py --mode part --name outlier_Larma --d_iter 30 --syn_outlier 20000 --syn_Lambda 50 --m 100 --syn_subsample 100000 --source dataset/armadillo-x.txt --train_type m --syn_m 100 --mass 100000 --beta 1.0 --Lambda 0.001 --sigma 1.0 --epoch 3001 --refine 1 --Refine_step 1 --vis 0 --save_pc 1')

# Large armadillo dataset
os.system('CUDA_VISIBLE_DEVICES=1 python main.py --mode partial --name partial_Larma --d_iter 30 --syn_outlier 0 --syn_Lambda 50 --syn_partial 0.7 --m 100 --syn_subsample 100000 --source dataset/armadillo-x.txt --train_type m --syn_m 100 --mass 40000 --beta 1.0 --Lambda 0.001 --sigma 1.0 --epoch 3001 --refine 1 --Refine_step 1 --vis 0 --normalize 0 --save_pc 1')
os.system('CUDA_VISIBLE_DEVICES=1 python main.py --mode partial --name partial_Larma --d_iter 30 --syn_outlier 0 --syn_Lambda 50 --syn_partial 0.7 --m 100 --syn_subsample 100000 --source dataset/armadillo-x.txt --train_type h --syn_m 100 --h 0.01 --beta 1.0 --Lambda 0.001 --sigma 1.0 --epoch 3001 --refine 1 --Refine_step 1 --vis 0 --normalize 0 --save_pc 1')

