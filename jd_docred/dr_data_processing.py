import numpy as np
import os
import json
import argparse
from tqdm import tqdm
import numpy as np

def preprocess(data_file_name, rel2id, max_length = 512, is_training = True, suffix=''):
    data = json.load(open(data_file_name))
    #### title, sents, labels, vertexSet
    print(len(data))
    sent_nums = []
    trip_nums = []
    evidence_nums = []
    word_nums = []
    example_triple_dict = {}
    relation_count_dict = {}
    for example_id, example in tqdm(enumerate(data)):
        # print(example)
        sent_nums.append(len(example['sents']))
        trip_nums.append(len(example['labels']))
        for exam_label in example['labels']:
            evidence_nums.append(len(exam_label['evidence']))
            h, r, t = exam_label['h'], exam_label['r'], exam_label['t']
            if r not in relation_count_dict:
                relation_count_dict[r] = 1
            else:
                relation_count_dict[r] = relation_count_dict[r] + 1
            exam_trip = (example_id, h, t)
            if exam_trip not in example_triple_dict:
                example_triple_dict[exam_trip] = 1
            else:
                example_triple_dict[exam_trip] = example_triple_dict[exam_trip] + 1
        total_word_num = sum([len(sent) for sent in example['sents']])
        word_nums.append(total_word_num)
        # break
    sent_num_array = np.array(sent_nums)
    trip_num_array = np.array(trip_nums)
    evid_num_array = np.array(evidence_nums)
    word_num_array = np.array(word_nums)
    example_ht_pair_num_array = np.array([value for key, value in example_triple_dict.items()])
    print(example_ht_pair_num_array.sum())
    # print('sent', sent_num_array.max(), sent_num_array.min(), sent_num_array.mean())
    # print('trip', trip_num_array.max(), trip_num_array.min(), trip_num_array.mean())
    # print('evid', evid_num_array.max(), evid_num_array.min(), evid_num_array.mean())
    # print('word', word_num_array.max(), word_num_array.min(), word_num_array.mean())
    # print('word_per {}'.format(np.percentile(word_num_array, [50, 75, 95, 97.5])))
    # print('total triples {}'.format(trip_num_array.sum()))
    # print('triple_per {}'.format(np.percentile(trip_num_array, [50, 75, 95, 97.5])))
    # print('sent_per {}'.format(np.percentile(sent_num_array, [50, 75, 95, 97.5])))
    # print('even_per {}'.format(np.percentile(evid_num_array, [50, 75, 95, 97.5])))
    #
    # (unique, counts) = np.unique(evid_num_array, return_counts=True)
    # uni_count_dict = dict(zip(unique, counts))
    # for key, value in uni_count_dict.items():
    #     print('{}\t{}'.format(key, value))

    # (unique, counts) = np.unique(example_ht_pair_num_array, return_counts=True)
    # uni_count_dict = dict(zip(unique, counts))
    # for key, value in uni_count_dict.items():
    #     print('{}\t{}'.format(key, value))
    for key, value in relation_count_dict.items():
        print(key, value)
    return

def docred2hotpotqa(data_file_name, rel2id, rel_infor, max_length = 512, is_training = True, suffix=''):
    docred_data = json.load(data_file_name)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='/Users/xjtuwgt/Downloads/DocRED', help='raw data path')
    parser.add_argument('--meta_path', type=str, default='/Users/xjtuwgt/Downloads/DocRED/DocRED_baseline_metadata', help='meta information path')
    parser.add_argument('--out_path', type=str, default=None, help='output data path')

    args = parser.parse_args()
    raw_data_path = args.raw_path
    out_path = args.out_path
    case_sensitive = False

    char_limit = 16
    train_distant_file_name = os.path.join(raw_data_path, 'train_distant.json')
    train_annotated_file_name = os.path.join(raw_data_path, 'train_annotated.json')
    dev_file_name = os.path.join(raw_data_path, 'dev.json')
    test_file_name = os.path.join(raw_data_path, 'test.json')

    relinfor = json.load(open(os.path.join(raw_data_path, 'rel_info.json'), "r"))
    print(len(relinfor))
    # print(relinfor)

    # print(train_distant_file_name)
    rel2id = json.load(open(os.path.join(args.meta_path, 'rel2id.json'), "r"))
    # print(rel2id)
    # for key, value in relinfor.items():
    #     print('{}\t{}\t{}'.format(key, value, rel2id[key]))
    ner2id = json.load(open(os.path.join(args.meta_path, 'ner2id.json'), "r"))
    # print(ner2id)
    char2id = json.load(open(os.path.join(args.meta_path, 'char2id.json'), "r"))
    # print(char2id)
    word2id = json.load(open(os.path.join(args.meta_path, 'word2id.json'), "r"))
    # print(len(word2id))

    ###+++++++++++++++++++++++++++++++++++++
    preprocess(data_file_name=train_annotated_file_name, rel2id=rel2id, is_training=True)

    # preprocess(data_file_name=dev_file_name, rel2id=rel2id, is_training=True)


