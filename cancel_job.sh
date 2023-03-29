#!/bin/bash


for job_id in {15857759..15857848};
    do
        scancel $job_id
done