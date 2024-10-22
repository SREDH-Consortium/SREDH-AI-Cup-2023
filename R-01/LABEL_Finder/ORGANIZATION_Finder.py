from LABEL_Finder.Finder import Finder
import re 
from LABEL_Finder.Finder_DICT import Finder_DICT
class ORGANIZATION_Finder(Finder):
    def __init__(self , dict_path = rf'dictionary\ORGANIZATION_dict_V3.txt' , force_dict_path = rf'dictionary\ORGANIZATION_force_dict.txt' ):# rf'dictionary\ORGANIZATION_force_dict.txt'
        super().__init__()
        self.dict_path = dict_path
        self.force_dict_path = force_dict_path
        self.PATTERN = [ 
        ]
        self.res_label = []
        self.ORG_KEYWORD = [
            'Services'      ,
            'Service'       ,
            'USA'           ,
            'Co'            ,
            'Ltd'           ,
            'Inc'           ,
            'Corp'          ,
            'Corporation'   ,
            'Company'       ,
            'Group'         ,
        ]
        
    
    # in dict maker 
    '''
    DELETE_WORDS = [
        'Co'       ,
        'Ltd'      ,
        'Inc'      ,
        'Corp'     ,
        'Corporation',
        'Company'   ,
        'Group'     ,
    ]
    '''
    # Corporation|Company|Corp|Co|Group|Ltd|Inc
    def find(self):
        dict_finder = Finder_DICT(self.dict_path)
        dict_finder.set_file(self.file)
        # dict_finder.dict_len_filter(5)
        self.PATTERN = dict_finder.find_all()
        for i in range(len(self.PATTERN)):
            self.PATTERN[i] = '(?i)' + '(' + '(?:The )?' + self.PATTERN[i] + '(?: (?:Services|Service))?' + '(?: (?:Corporation|Company|Corp|Co|Group|Ltd|Inc))?' +')'
        self.res_label = self.re_find(self.PATTERN)
        # filter if len < 9 del and all ORG_KEYWORD not in it
        
        new_label = []
        for lb in self.res_label:
            s,e,w = lb
            if len(w) < 9:
                have_keyword = False
                for key in self.ORG_KEYWORD:
                    if key in w:
                        have_keyword = True
                        break 
                if have_keyword:
                    new_label.append(lb)
            else:
                new_label.append(lb)
        self.res_label = new_label
        
        self.res_label = self.del_same(self.res_label)
        self.res_label = self.remove_overlamp(self.res_label)
        
        ### force  add lb by 
        ## add label before self.res_label
        if self.force_dict_path!=None:
            dict_finder = Finder_DICT(self.force_dict_path)
            dict_finder.set_file(self.file)
            self.PATTERN = dict_finder.find_all()
            for i in range(len(self.PATTERN)):
                self.PATTERN[i] = r'(' + self.PATTERN[i] + r')'
            self.res_label = self.re_find(self.PATTERN) + self.res_label
            self.res_label = self.del_same(self.res_label)
            self.res_label = self.remove_overlamp(self.res_label)
                        
        return self.res_label
    



if __name__=='__main__':
    # print now dir 
    import os
    print(os.getcwd())
    
    finder = ORGANIZATION_Finder()
    
    finder.set_file(rf'C:\Users\rui\Desktop\First_Phase_ReleaseCorrection\First_Phase_Release(Correction)\Validation_Release\904.txt')
    
    res = finder.find()
    
    print(res)
        
    

    #finder.set_file(rf'.\data\file\10.txt')
    
    
    #print(finder.find() )