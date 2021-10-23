import numpy as np
from Bio import SeqIO
def AA_ONE_HOT(AA):
    one_hot_dict = {'A': [0,1,0],
            'T': [1,0,0],
            'C': [0,0,0],
            'G': [0,0,1],
            'N': [0.5,0.5,0.5]
            }

    coding_arr = np.zeros((len(AA),3),dtype=float)

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
main_list=neg_list+pos_list
main_arr=np.zeros((len(main_list),41*3))
for i in range(len(main_list)):
    main_arr[i]=AA_ONE_HOT(main_list[i])
np.save('ONEHOT',main_arr)
