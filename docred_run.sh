#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files
DATA_ROOT=./data/docred/DocRED


# 0. docred
docred() {
    INPUTS=("dev.json;dev" "test.json;test" "train_annotated.json;train")
    for input in ${INPUTS[*]}; do
      INPUT_FILE=$(echo $input | cut -d ";" -f 1)
      DATA_TYPE=$(echo $input | cut -d ";" -f 2)

      echo "Processing input_file: ${INPUT_FILE}"

      INPUT_FILE=$DATA_ROOT/$INPUT_FILE
      OUTPUT_PROCESSED=$DATA_ROOT/data_processed/$DATA_TYPE
      OUTPUT_FEAT=$DATA_ROOT/data_feat/$DATA_TYPE

      [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
      [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT
    done
}

preprocess() {
      echo "1. data statistics"
      python jd_docred/dr_data_processing.py --raw_path $DATA_ROOT --meta_path $DATA_ROOT/DocRED_baseline_metadata --out_path $DATA_ROOT/data_processed
}

#for proc in "docred" "preprocess"
for proc in "preprocess"
do
    $proc
done
