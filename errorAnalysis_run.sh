#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files, transformers 3.3.0
DATA_ROOT=./data/
LONG_FORMER_ROOT=allenai
SELECTEED_DOC_NUM=4


PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only

# define precached BERT MODEL path
ROBERTA_LARGE=$DATA_ROOT/models/pretrained/roberta-large

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/hgntransformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

mkdir -p $DATA_ROOT/models/pretrained_cache

preprocess() {
    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE
        PRED_OUTPUT=$DATA_ROOT/outputs

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT

        echo "Error Analysis"
#        python errorAnalysis/predictionAnalysis.py --raw_data $INPUT_FILE --input_dir $OUTPUT_FEAT --pred_dir $PRED_OUTPUT --output_dir $OUTPUT_FEAT --model_type roberta --model_name_or_path roberta-large
        python errorAnalysis/predictionAnalysis.py --raw_data $INPUT_FILE --input_dir $OUTPUT_FEAT --pred_dir $PRED_OUTPUT --output_dir $OUTPUT_FEAT --model_type albert --model_name_or_path albert-xxlarge-v2
    done

}

for proc in "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done
