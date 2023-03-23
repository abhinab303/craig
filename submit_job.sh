#!/bin/bash
base_job_name="wrmch"
job_file="the_job.sh"
identifier_name="wrmch"
dir="op_"$identifier_name
mkdir -p $dir


methods="TD"
ss_sizes={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}

job_name=$base_job_name-"EE"
out_file=$dir/$job_name.out
error_file=$dir/$job_name.err

ss_list=(0.1)
rn_list=(0 1 2)

for ss_size in "${ss_list[@]}";
do
    for rn in "${rn_list[@]}";
    do
        export ss_size 
        export rn
        job_name=$base_job_name-$rn-$ss_size
        out_file=$dir/$job_name.out
        error_file=$dir/$job_name.err

        echo $ss_size $rn------------------------------------------------------------------
        sbatch -J $job_name -o $out_file -e $error_file $job_file
    done
done

# sbatch -J $job_name -o $out_file -e $error_file $job_file