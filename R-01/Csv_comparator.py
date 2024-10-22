# the class can compare two csv files and return the difference between them
# it can also return the difference between two data frames
# it can show the difference at terminal with different color for each difference
# it print each labe with different color
# and it can calculate the fl score with the answer csv file

'''
True Positive（簡稱 TP）：參賽團隊系統所辨識的 PHI 在文本中的起始與結束位置及其預測的類別完全符合人工標注的 PHI 對應的三個屬性；
False Positive（簡稱 FP）：系統辨識的 PHI 的任何一種屬性（起始、結尾或類別）不符合任何人工標注的 PHI 結果；
False Negative（簡稱 FN）：人工標注的 PHI 找不到完全匹配的系統辨識結果。
'''


import pandas as pd
import re
import csv
from termcolor import colored

class Csv_comparator:
    def __init__(self , my_ans_path , target_path , specify_label = None , ignore_time = False):
        self.col_name = ['file','label','start','end','content','time']
        #define data type
        self.col_dtype = {'file':str,'label':str,'start':int,'end':int,'content':str,'time':str}
        self.my_ans_df = pd.read_csv(my_ans_path,sep='\t',names=self.col_name , dtype=self.col_dtype)
        self.target_df = pd.read_csv(target_path,sep='\t',names=self.col_name , dtype=self.col_dtype)
        self.correct_df = pd.DataFrame(columns=self.col_name)
        if ignore_time==True:
            self.my_ans_df.drop(columns=['time'],inplace=True)
            self.target_df.drop(columns=['time'],inplace=True)
            
                
        self.TP = 0 #完全符合人工標注
        self.FP = 0 #不符合任何人工標注的
        self.FN = 0 #人工標注的 PHI 找不到完全匹配的系統辨識結果
        
        #Recall(召回率) = TP/(TP+FN)
        #Precision(準確率) = TP/(TP+FP)
        #F1-score = 2 * Precision * Recall / (Precision + Recall)
        
        
        if specify_label != None:
            if type(specify_label) == str:
                specify_label = [specify_label]
            elif type(specify_label) == list:
                pass
            self.my_ans_df = self.my_ans_df.loc[self.my_ans_df['label'].isin(specify_label)]
            self.target_df = self.target_df.loc[self.target_df['label'].isin(specify_label)]
        
        
    def calc_f1_score(self):
        Recall = self.TP/(self.TP+self.FN)
        Precision = self.TP/(self.TP+self.FP)
        try:
            F1_score = 2 * Precision * Recall / (Precision + Recall)
        except:
            F1_score = 0
        #F1_score = 2 * Precision * Recall / (Precision + Recall)
        print("Recall:",Recall)
        print("Precision:",Precision)
        print("F1_score:",F1_score)
        
    def compare(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.target_df['drop'] = False
        self.my_ans_df['drop'] = False
        self.correct_df = pd.DataFrame(columns=self.col_name)
        
        for index1,row1 in self.target_df.iterrows():
            for index2,row2 in self.my_ans_df.loc[self.my_ans_df['file']==row1['file']].iterrows():
                
                if row1.equals(row2):
                # if row1.str==row2.str:
                    #print("eq")
                    self.TP = self.TP + 1
                    self.target_df.loc[index1,'drop'] = True
                    self.my_ans_df.loc[index2,'drop'] = True
                    # add to correct_df
                    self.correct_df.loc[len(self.correct_df)] = row1
                    
                    break
                    # self.target_df.drop(index1,inplace=True )
                    # self.my_ans_df.drop(index2,inplace=True )
                
        # drop if drop == true
        self.target_df.drop(self.target_df[self.target_df['drop']==True].index,inplace=True)
        self.my_ans_df.drop(self.my_ans_df[self.my_ans_df['drop']==True].index,inplace=True)

        #drop 'drop' column
        self.target_df.drop(columns=['drop'],inplace=True)
        self.my_ans_df.drop(columns=['drop'],inplace=True)
        
        self.target_df.reset_index(drop=True, inplace=True)
        self.my_ans_df.reset_index(drop=True, inplace=True)
        
        self.FP = self.my_ans_df.index.size
        self.FN = self.target_df.index.size
                   
        
    def print_res(self):
        print("\n\nmy_ans_df:" ,'your wrong answer')
        print(self.my_ans_df)
        print("\n\ntarget_df:",' you dont found the following')
        #  print end of  10 data
        print(self.target_df)
    def save_res(self , dir=''):
        self.my_ans_df.to_csv('.output/your wrong answer.csv' , sep='\t' , header=False)
        self.target_df.to_csv('.output/you dont found the following.csv' , sep='\t' , header=False)
        self.correct_df.to_csv('.output/correct.csv' , sep='\t' , header=False)
        
    
        

if __name__ == "__main__":
    # my_ans_path = rf'test_data\csv\ans1.csv'
    # target_path = rf'test_data\csv\ans2.csv'
    
    my_ans_path = rf'test.csv'
    target_path = rf'data\answer.txt'
    #target_path = rf'test2.csv'
    # 
    
    comparator = Csv_comparator(my_ans_path,target_path ,  specify_label = None)
    comparator.compare()
    comparator.print_res()
    comparator.calc_f1_score()
    
    comparator.save_res()
        
        
    



