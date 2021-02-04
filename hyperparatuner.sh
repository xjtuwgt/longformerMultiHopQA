#!/usr/bin/env bash
JOBS_PATH=athgn_jobs
LOGS_PATH=athgn_logs
for ENTRY in "${JOBS_PATH}"/*.sh; do
  chmod +x $ENTRY
  FILE_NAME="$(basename "$ENTRY")"
  echo $FILE_NAME
  /mnt/cephfs2/asr/users/ming.tu/software/kaldi/egs/wsj/s5/utils/queue.pl -q g.q -l gpu=4 $LOGS_PATH/$FILE_NAME.log $ENTRY &
  sleep 20
done