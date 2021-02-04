#!/usr/bin/env bash
#!/bin/sh
CONFIG_PATH=configs/athgn
LOGS_PATH=athgn_logs
for ENTRY in "${CONFIG_PATH}"/*.json; do
  echo "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 jdattrain.py --config_file configs/athgn/${ENTRY}" > "jobs/run_${ENTRY}.sh"
  chmod +x "jobs/run_${ENRTY}.sh"
#  /mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g.q -l gpu=4 $LOGS_PATH/$FILE_NAME.log $ENTRY &
#  sleep 20
done