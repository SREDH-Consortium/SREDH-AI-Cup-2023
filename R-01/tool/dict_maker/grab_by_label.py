
ALL_ANSWER_PATH = [
    rf'data\First_Phase\answer.txt'                 ,
    rf'data\First_Phase_Validation\answer.txt'      ,
    rf'data\Second_Phase\answer.txt'                ,
]

# GRAB_LABEL = 'ORGANIZATION'
# GRAB_LABEL = 'DEPARTMENT'
GRAB_LABEL = 'CITY'

import pandas as pd 
import numpy as np


def csv_reader(file_path):
    names=['file_name','label','start','end','content','time']
    df = pd.read_csv(file_path , sep='\t', header=None , names=names)
    return df


if __name__=='__main__':
    all_df = []
    for path in ALL_ANSWER_PATH:
        df = csv_reader(path)
        all_df.append(df)
    df = pd.concat(all_df)
    
    # df.to_csv(rf'./output/merge_ans.txt', sep='\t', header=None, index=False)
    label_df = df[df['label']==GRAB_LABEL]
    # label_df.to_csv(rf'DEPARTMENT_label.txt', sep='\t', header=None, index=False)
    # print(label_df)
    r = label_df['content'].unique()
    print(r)
    print(len(r))
    
    # write r to file
    with open(rf'tool/dict_maker/force_dict.txt', 'w', encoding='utf-8') as f:
        for line in r:
            f.write(line+'\n')
            
    
    # with open(rf'./output/force_dict.txt', 'r', encoding='utf-8') as f:
        
    # df different  content to list
    
    
    
    