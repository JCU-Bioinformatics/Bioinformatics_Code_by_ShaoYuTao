import numpy as np
from Bio import SeqIO
def AA_ONE_HOT(AA):
    one_hot_dict = {'A': [1],
            'T': [2],
            'C': [3],
            'G': [4],
            'N': [5]
            }

    coding_arr = np.zeros((len(AA),1),dtype=float)

    for i in range(len(AA)):

        coding_arr[i] =  one_hot_dict[AA[i]]

    return coding_arr.flatten()
neg_list=[]
for seqrecord in SeqIO.parse('negative.fasta','fasta'):
    seq=str(seqrecord.seq)
    neg_list.append(seq)
pos_list=[]
for seqrecord in SeqIO.parse('positive.fasta','fasta'):
    seq = str(seqrecord.seq)
    pos_list.append(seq)

main_listpos=pos_list
main_listneg=neg_list
main_list=neg_list+pos_list

main_arrpos=np.zeros((len(main_listpos),41*1))
main_arrneg=np.zeros((len(main_listneg),41*1))
main_arr=np.zeros((len(main_list),41*1))

for i in range(len(main_listpos)):
    main_arrpos[i]=AA_ONE_HOT(main_listpos[i])
for i in range(1, 6):
    print("pos{}: {}".format(i, np.sum(main_arrpos == i)))

for i in range(len(main_listneg)):
    main_arrneg[i] = AA_ONE_HOT(main_listneg[i])
for i in range(1, 6):
    print("neg{}: {}".format(i, np.sum(main_arrneg == i)))

for i in range(len(main_list)):
    main_arr[i] = AA_ONE_HOT(main_list[i])
for i in range(1, 6):
    print("all{}: {}".format(i, np.sum(main_arr == i)))