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
#     # Input: INPUT_FILE, enwiki_ner.db
#     # Output: doc_link_ner.json
#        python scripts/1_extract_db.py $INPUT_FILE $DATA_ROOT/knowledge/enwiki_ner.db $OUTPUT_PROCESSED/doc_link_ner.json
#
#        echo "2. Extract NER for Question and Context"
#        # Input: doc_link_ner.json
#        # Output: ner.json
#        python scripts/2_extract_ner.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json
#
#        echo "3. Paragraph ranking"
#        # Output: para_ranking.json
#        python scripts/3_prepare_para_sel.py $INPUT_FILE $OUTPUT_PROCESSED/hotpot_ss_$DATA_TYPE.csv
#
#        # switch to RoBERTa for final leaderboard
#        python scripts/3_paragraph_ranking.py --data_dir $OUTPUT_PROCESSED --eval_ckpt $DATA_ROOT/models/finetuned/PS/pytorch_model.bin --raw_data $INPUT_FILE --input_data $OUTPUT_PROCESSED/hotpot_ss_$DATA_TYPE.csv --model_name_or_path roberta-large --model_type roberta --max_seq_length 256 --per_gpu_eval_batch_size 128 --fp16

#        echo "4. MultiHop Paragraph Selection"
#        # Input: $INPUT_FILE, doc_link_ner.json,  ner.json, para_ranking.json
#        # Output: multihop_para.json
#        python scripts/4_multihop_ps.py $INPUT_FILE $OUTPUT_PROCESSED/doc_link_ner.json $OUTPUT_PROCESSED/ner.json $OUTPUT_PROCESSED/para_ranking.json $OUTPUT_PROCESSED/multihop_para.json
#
#        echo "5. Dump features"
#        python scripts/5_dump_features.py --para_path $OUTPUT_PROCESSED/multihop_para.json --full_data $INPUT_FILE --model_name_or_path roberta-large --ner_path $OUTPUT_PROCESSED/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json
##        python scripts/5_dump_features.py --para_path $OUTPUT_PROCESSED/multihop_para.json --full_data $INPUT_FILE --model_name_or_path albert-xxlarge-v2 --do_lower_case --ner_path $OUTPUT_PROCESSED/ner.json --model_type albert --tokenizer_name albert-xxlarge-v2 --output_dir $OUTPUT_FEAT --doc_link_ner $OUTPUT_PROCESSED/doc_link_ner.json
#
##        echo "6. Test dumped features"
#        #python scripts/6_test_features.py --full_data $INPUT_FILE --input_dir $OUTPUT_FEAT --output_dir $OUTPUT_FEAT --model_type roberta --model_name_or_path roberta-large
#    done

}

for proc in "docred" "preprocess"
do
    $proc
done
