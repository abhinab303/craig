#!/bin/bash


for job_id in {15843283..15843291};
    do
        scancel $job_id
done