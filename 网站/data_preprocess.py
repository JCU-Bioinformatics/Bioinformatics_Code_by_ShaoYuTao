#!usr/bin/env python
# -*- coding:utf-8 -*-
# Author:zhanglei Time:2020/1/3 12:58



try:
    import tensorflow.python.keras as keras
except:
    import tensorflow.keras as keras


import os
import numpy as np
import pandas as pd
import time
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error和warining信息 3 只显示error信息
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这一行注释掉就是使用cpu，不注释就是使用gpu


def AA_ONE_HOT(AA):
    one_hot_dict = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    coding_arr = np.zeros((len(AA), 4), dtype=float)

    for m in range(len(AA)):
        coding_arr[m] = one_hot_dict[AA[m]]

    return coding_arr


def seq_analyze(strand_where, original_file):
    all_record = []
    count = 0
    count_number=[]
    description = []
    if strand_where == 'forward strand':
        with open(r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/fasta_data/' + str(Program_name_all) + r'/Fasta_files/pre_seq.fasta', 'w') as sr:
            for seq_load_1 in SeqIO.parse(original_file, 'fasta'):
                number = 0
                seq_cd = len(seq_load_1.seq)
                record_num = []
                new_seq = str(seq_load_1.seq)[-20:][::-1] + str(seq_load_1.seq) + str(seq_load_1.seq)[:20][::-1]
                for i in range(seq_cd):
                    if new_seq[20 + i] == 'C':
                        record_num.append(i + 1)
                        sr.write('>' + str(seq_load_1.description) + '\n')
                        sr.write(new_seq[i:i + 41] + '\n')
                        all_record.append(new_seq[i:i + 41])
                        description.append(str(seq_load_1.description))
                        number+=1
                        count_number.append(number)
                count += 1
    elif strand_where == 'reverse strand':
        with open(r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/fasta_data/' + str(Program_name_all) + r'/Fasta_files/pre_seq.fasta', 'w') as sr:
            for seq_load_1 in SeqIO.parse(original_file, 'fasta'):
                number = 0
                seq_cd = len(seq_load_1.seq)
                record_num = []
                new_seq = str(seq_load_1.seq)[-20:][::-1] + str(seq_load_1.seq) + str(seq_load_1.seq)[:20][::-1]
                for i in range(seq_cd):
                    if new_seq[20 + i] == 'G':
                        record_num.append(i + 1)
                        sr.write('>' + str(seq_load_1.description) + '\n')
                        sr.write(str(Seq(new_seq[i:i + 41], IUPAC.ambiguous_dna).reverse_complement()) + '\n')
                        all_record.append(str(Seq(new_seq[i:i + 41])))
                        description.append(str(seq_load_1.description))
                        number+=1
                        count_number.append(number)
                count += 1
    return count, description, all_record, count_number


def getfasta_file(seq_strand, program_names, original_file):
    global Program_names, Program_name_all
    program_time = time.strftime("-%b%d-%H%M%S", time.localtime())
    Program_names = program_names
    Seq_strand = seq_strand
    original_file = original_file
    Program_name_all = str(program_names) + str(program_time)
    # print(Program_name_all)
    fasta_files_dir = r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/fasta_data/' + str(Program_name_all) + r'/Fasta_files'

    fasta_file_dir_Exists = os.path.exists(fasta_files_dir)

    if not fasta_file_dir_Exists:
        # 如果不存在则创建目录
        os.makedirs(fasta_files_dir)
    count, description, all_record, count_number= seq_analyze(Seq_strand, original_file)
    fasta_pre_lable = []
    oo = 0
    count_seq = 0
    for my_aa_count in SeqIO.parse(r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/funsion_process/pre_seq.fasta', 'fasta'):
        count_seq += 1
    a = np.zeros((count_seq, 41, 4))
    aa = a.copy()
    for my_aa in SeqIO.parse(r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/funsion_process/pre_seq.fasta', 'fasta'):
        AA = str(my_aa.seq)
        aa[oo] = AA_ONE_HOT(AA)
        oo += 1
    sample = aa.reshape(count_seq, -1)
    k = 0
    y_predict_class = np.zeros(count_seq, dtype=np.float64)
    y_predict_my = np.zeros((count_seq, 2), dtype=np.float64)
    for i in range(11):
        model_name = r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/funsion_model/funsion_model' + str(k + 1) + '.h5'
        model = keras.models.load_model(model_name)
        y_predict = model.predict(sample.reshape(count_seq, -1))
        y_predict_my += y_predict
        y_predict_class += np.argmax(y_predict, axis=1)
        k += 1
        del model
    negative = (y_predict_my / 11)[:, 0]
    positive = (y_predict_my / 11)[:, 1]
    list = np.zeros(count_seq, dtype=np.float64)
    for i, num in enumerate(y_predict_class):
        if num >= 11:
            list[i] = 1
        elif num < 11:
            list[i] = 0
    stand = 0
    for o in range(count_seq):
        content = {
            'seq_original': description[o],
            'original_seq': all_record[o],
            'methy_site_number': count_number[o],
            'Pre_lable': list[o],
            'negative_probably': negative[o],
            'positive_probably': positive[o]
        }
        stand += len(all_record[o])
        fasta_pre_lable.append(content)
    route = r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/fasta_data/' + str(Program_name_all) + r'/Fasta_files/' + Program_name_all + '.csv'
    pd.DataFrame(fasta_pre_lable).to_csv(route)
    result_load = r'/home/A1508_xgf/jci_webserver/iPromoter-5mC/save_model/fasta_data/' + str(Program_name_all) + r'/Fasta_files/' + Program_name_all + '.csv'
    original_file_path = Program_name_all + '.csv'
    return result_load, original_file_path
