#!/bin/bash
base_job_name="mnist_craig_el2n_shuffle_init"
job_file="the_job.sh"
identifier_name="mnist_craig_el2n_shuffle_init"
dir="rc_out/op_"$identifier_name
mkdir -p $dir


methods="TD"
ss_sizes={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}

job_name=$base_job_name-"EE"
out_file=$dir/$job_name.out
error_file=$dir/$job_name.err

# ss_list=(0.5)
# rn_list=(0 1 2)
# tau_list=(0.1 0.3 0.5 0.7 0.9 1)
# lam_list=(30 50 90 200 400 4000) 

ss_list=(0.5)
rn_list=(5)
tau_list=(0.1 0.3 0.5 0.7 0.9)
lam_list=(5 10 15 30 70 200) 

# ss_list=(0.5)
# rn_list=(5)
# tau_list=(0.1 0.3 0.5 0.7 0.9 1)
# lam_list=(5) 

for ss_size in "${ss_list[@]}";
do
    for tau in "${tau_list[@]}";
    do
        for lam in "${lam_list[@]}"
        do
            for rn in "${rn_list[@]}";
            do
                export ss_size 
                export rn
                export tau
                export lam
                export base_job_name
                job_name=$base_job_name-$rn-$ss_size-$tau-$lam
                out_file=$dir/$job_name.out
                error_file=$dir/$job_name.err
                
                export job_name
                echo $ss_size $rn $tau $lam------------------------------------------------------------------
                sbatch -J $job_name -o $out_file -e $error_file $job_file
            done
        done
    done
done

# sbatch -J $job_name -o $out_file -e $error_file $job_file