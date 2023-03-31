#!/bin/bash -lT


#SBATCH --job-name=EDL_RES_1
#SBATCH --time 00-4:00:00
#SBATCH --account cisc-896 --partition tier3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6g
#SBATCH --gres=gpu:1


#conda activate sn_coreset
conda activate craig_mnist

#python train_resnet_el2n.py -s $ss_size -w -b 128 -g --smtk 0 -run $rn -jb $job_name -bj $base_job_name -tau $rp
#python train_resnet.py -s $ss_size -w -b 128 -g --smtk 0 -run $rn -jb $job_name -bj $base_job_name
#python train_resnet.py -s $ss_size -w -b 246 -run $rn -rand $rp
python mnist.py --tau $tau --lam $lam --jb $job_name --bj $base_job_name
# python train_resnet_craig_el2n.py -s $ss_size -w -b 128 -g --smtk 0 -run $rn -jb $job_name -bj $base_job_name -tau $tau -lam $lam

# python train_resnet_craig_el2n.py -s 0.5 -w -b 128 -g --smtk 0 -run 5 -jb 5 -bj 5 -tau 0.5 -lam 5