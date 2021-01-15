import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import os
import pandas as pd
from pandas import DataFrame
from time import time
import logging
from datetime import date, datetime

hotpot_path = '../data/hotpotqa/'
hotpot_path = os.path.abspath(hotpot_path)
print('abs_path: {}'.format(hotpot_path))
hotpot_train_data = 'hotpot_train_v1.1.json'  # _id;answer;question;supporting_facts;context;type;level
hotpot_dev_fullwiki = 'hotpot_dev_fullwiki_v1.json'  # _id;answer;question;supporting_facts;context;type;level
hotpot_test_fullwiki = 'hotpot_test_fullwiki_v1.json'  # _id; question; context
hotpot_dev_distractor = 'hotpot_dev_distractor_v1.json'  # _id;answer;question;supporting_facts;context;type;level
gold_hotpot_dev_distractor = 'gold_hotpot_dev_distractor_v1.json'
hotpot_dev_retrieval_distractor = 'all_results4.pkl'

def loadJSONData(PATH, json_fileName)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(os.path.join(PATH, json_fileName), orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def loadPickleData(PATH, pickle_fileName):
    start_time = time()
    data = pd.read_pickle(os.path.join(PATH, pickle_fileName))
    print('Loading {} in {:.4f} seconds'.format(data.shape, time() - start_time))
    return data

def HOTPOT_Retrieval_Data(path=hotpot_path, reverse=True):
    data = loadPickleData(PATH=path, pickle_fileName=hotpot_dev_retrieval_distractor)
    data = data.transpose((1,0,2))
    records_num, model_num, doc_num = data.shape
    ir_res = []
    for row_idx in range(records_num):
        row_i = data[row_idx]
        row_ir_dict_i = dict()
        for model_idx in range(model_num):
            if reverse:
                row_ir_dict_i['model_{}'.format(model_idx+1)] = row_i[model_idx][::-1]
            else:
                row_ir_dict_i['model_{}'.format(model_idx + 1)] = row_i[model_idx]
        ir_res.append(row_ir_dict_i)
    df = pd.DataFrame(ir_res)
    return df

def HOTPOT_TrainData(path=hotpot_path):
    data = loadJSONData(PATH=path, json_fileName=hotpot_train_data)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_DevData_FullWiki(path=hotpot_path):
    data = loadJSONData(PATH=path, json_fileName=hotpot_dev_fullwiki)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_DevData_Distractor(path=hotpot_path):
    print('*'*75 + '\n' + path + '\n' + '*' *75)
    data = loadJSONData(PATH=path, json_fileName=hotpot_dev_distractor)
    column_names = [col for col in data.columns]
    return data, column_names

def GOLD_HOTPOT_DevData_Distractor(path=hotpot_path):
    data = loadJSONData(PATH=path, json_fileName=gold_hotpot_dev_distractor)
    column_names = [col for col in data.columns]
    return data, column_names

def HOTPOT_Test_FullWiki(path=hotpot_path):
    data = loadJSONData(PATH=path, json_fileName=hotpot_test_fullwiki)
    column_names = [col for col in data.columns]
    return data, column_names

def save_data_frame_to_json(df: DataFrame, file_name: str):
    df.to_json(file_name, orient='records')
    print('Save {} data in json file'.format(df.shape))
########################################################################################################################

def create_dir_if_not_exist(save_path, sub_folder=None):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
        logging.info('Create folder at {}'.format(save_path))
        if sub_folder is not None:
            os.makedirs(os.path.join(save_path, sub_folder))
            print('Create folder {} under {}'.format(sub_folder, save_path))
    if os.path.exists(save_path):
        sub_folder_path = os.path.join(save_path, sub_folder)
        if sub_folder is not None and not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)
            print('Create folder {} under {}'.format(sub_folder, save_path))

def get_date_time():
    today = date.today()
    str_today = today.strftime('%b_%d_%Y')
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    date_time_str = str_today + '_' + current_time
    return date_time_str

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    date_time_str = get_date_time()
    log_dir = os.path.join(args.log_path, args.log_name)
    log_file = os.path.join(log_dir, 'console_' + args.log_name + '_' + date_time_str + '.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

