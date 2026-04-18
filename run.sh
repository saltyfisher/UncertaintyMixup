# python train.py --uncertaintymixup --matting --dataset chestct --superpixel --alphalabel --num_trials 1 --save_results

# python train.py --uncertaintymixup --matting --dataset breakhis --magnification 40 --superpixel --alphalabel --num_trials 1 --save_results

# python train.py --uncertaintymixup --matting --dataset breakhis --magnification 100 --superpixel --alphalabel --num_trials 1 --save_results

# python train.py --uncertaintymixup --matting --dataset breakhis --magnification 200 --superpixel --alphalabel --num_trials 1 --save_results

# python train.py --uncertaintymixup --matting --dataset breakhis --magnification 400 --superpixel --alphalabel --num_trials 1 --save_results
##########################################################
python train.py --dataset chestct --magnification 400 --strategy mixup --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 40 --strategy mixup --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 100 --strategy mixup --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 200 --strategy mixup --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 400 --strategy mixup --num_trials 1 --save_results
##########################################################
python train.py --dataset chestct --magnification 400 --strategy cutmixrand --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 40 --strategy cutmixrand --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 100 --strategy cutmixrand --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 200 --strategy cutmixrand --num_trials 1 --save_results

python train.py --dataset breakhis --magnification 400 --strategy cutmixrand --num_trials 1 --save_results