#!/bin/bash

#export PYTHONPATH='../..:../../lib/fairseq'

python -u ../../metacomp.py \
  > run_noatt.out \
  2> run.err
