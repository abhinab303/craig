#!/bin/bash


for job_id in {15855764..15855817};
    do
        scancel $job_id
done