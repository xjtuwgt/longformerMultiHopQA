#!/usr/bin/env bash

# DEFINE data related (please make changes according to your configurations)
# DATA ROOT folder where you put data files, transformers 3.3.0
DATA_ROOT=./data/
LONG_FORMER_ROOT=allenai
SELECTEED_DOC_NUM=4


PROCS=${1:-"download"} # define the processes you want to run, e.g. "download,preprocess,train" or "preprocess" only

# define precached BERT MODEL path
# ROBERTA_LARGE=$DATA_ROOT/models/pretrained/roberta-large
# pip install -U spacy
# python -m spacy download en_core_web_lg

# Add current pwd to PYTHONPATH
export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:$DIR_TMP:$DIR_TMP/hgntransformers
export PYTORCH_PRETRAINED_BERT_CACHE=$DATA_ROOT/models/pretrained_cache

mkdir -p $DATA_ROOT/models/pretrained_cache

# 0. Build Database from Wikipedia
download() {
    [[ -d $DATA_ROOT ]] || mkdir -p $DATA_ROOT/dataset/data_raw; mkdir -p $DATA_ROOT/knowledge

    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    wget -P $DATA_ROOT/dataset/data_raw http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
    wget -P $DATA_ROOT/dataset/data_raw/ http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json
    if [[ ! -f $DATA_ROOT/knowledge/enwiki_ner.db ]]; then
        wget -P $DATA_ROOT/knowledge/ https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
        tar -xjvf $DATA_ROOT/knowledge/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -C $DATA_ROOT/knowledge
        # Required: DrQA and Spacy
        python scripts/0_build_db.py $DATA_ROOT/knowledge/enwiki-20171001-pages-meta-current-withlinks-abstracts $DATA_ROOT/knowledge/enwiki_ner.db
    fi
}

preprocess() {
    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor" "hotpot_train_v1.1.json;train")
#    INPUTS=("hotpot_dev_distractor_v1.json;dev_distractor")
    for input in ${INPUTS[*]}; do
        INPUT_FILE=$(echo $input | cut -d ";" -f 1)
        DATA_TYPE=$(echo $input | cut -d ";" -f 2)

        echo "Processing input_file: ${INPUT_FILE}"

        INPUT_FILE=$DATA_ROOT/dataset/data_raw/$INPUT_FILE
        OUTPUT_PROCESSED=$DATA_ROOT/dataset/data_processed/$DATA_TYPE
        OUTPUT_FEAT=$DATA_ROOT/dataset/data_feat/$DATA_TYPE

        [[ -d $OUTPUT_PROCESSED ]] || mkdir -p $OUTPUT_PROCESSED
        [[ -d $OUTPUT_FEAT ]] || mkdir -p $OUTPUT_FEAT

        echo "1. Extract Wiki Link & NER from DB"
        # Input: INPUT_FILE, enwiki_ner.db
        # Output: doc_link_ner.json
        python scripts/1_extract_db.py $INPUT_FILE $DATA_ROOT/knowledge/enwiki_ner.db $OUTPUT_PROCESSED/doc_link_ner.json

        echo "2. Extract NER for Question and Context"
        # Input: doc_link_ner.json
        # Output: ner.json
        python scripts/2_extract_ner.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json

        echo "3. Paragraph ranking (1)"
        # Output: para_ranking.json
        python scripts/3_prepare_para_sel.py $INPUT_FILE $OUTPUT_PROCESSED/hotpot_ss_$DATA_TYPE.csv

        echo "3. Paragraph ranking (2): longformer retrieval data preprocess"
        # Output: para_ranking.json
        python longformerscripts/3_longformer_prepare_para_sel.py $INPUT_FILE $OUTPUT_PROCESSED/para_ir_combined.json

        echo "3. Paragraph ranking (3): longformer retrieval ranking scores"

        # switch to Longformer for final leaderboard
        python longformerscripts/3_longformer_paragraph_ranking.py --data_dir $OUTPUT_PROCESSED --eval_ckpt $DATA_ROOT/models/finetuned/PS/longformer_pytorchlighting_model.ckpt --raw_data $INPUT_FILE --input_data $OUTPUT_PROCESSED/para_ir_combined.json

        echo "4. MultiHop Paragraph Selection (4)"
        # Input: $INPUT_FILE, doc_link_ner.json,  ner.json, long_para_ranking.json
        # Output: long_multihop_para.json
        python longformerscripts/4_longformer_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED/long_para_ranking.json $OUTPUT_PROCESSED/long_multihop_para.json $SELECTEED_DOC_NUM

        echo "5. Dump features (5)"
        python longformerscripts/5_ext_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path roberta-large --ner_path $OUTPUT_PROCESSED/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --max_para_num $SELECTEED_DOC_NUM
#        python longformerscripts/5_ext_dump_features.py --para_path $OUTPUT_PROCESSED/long_multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json --max_para_num $SELECTEED_DOC_NUM

    done

}

for proc in "download" "preprocess"
do
    if [[ ${PROCS:-"download"} =~ $proc ]]; then
        $proc
    fi
done
