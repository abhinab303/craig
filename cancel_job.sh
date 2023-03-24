#!/bin/bash


for job_id in {15844222..15844229};
    do
        scancel $job_id
done