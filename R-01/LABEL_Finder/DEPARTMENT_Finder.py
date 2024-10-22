from LABEL_Finder.Finder import Finder
import re 
from LABEL_Finder.Finder_DICT import Finder_DICT

class DEPARTMENT_Finder(Finder):
    def __init__(self , dict_path = rf'dictionary\DEPARTMENT_force_dict.txt'):
        super().__init__()
        self.dict_path = dict_path
        self.PATTERN = [ #PATIENT
            # r'(Cure For Life Foundation Neuro-oncology Laboratory Adult Cancer)'        ,
            
            #11/23 V1
            r"Location: *([A-Za-z0-9\']+(?: +[A-Za-z0-9\']+)*) *-"          ,

            # V2
            # r"Location: *((?:PERI-|Cure For Life Foundation Neuro-)?[A-Za-z0-9\']+(?: +[A-Za-z0-9\']+)*) *-"          ,
        
        ]
        self.dict_pattern = []
        self.res_label = []
        
        
    def find(self):
        # self.res_label = self.re_find(self.PATTERN)
        # self.res_label = self.del_same(self.res_label)
        
        
        if self.dict_path != None:
            dict_finder = Finder_DICT(self.dict_path)
            dict_finder.set_file(self.file)
            # dict_finder.dict_len_filter(4)
            self.dict_pattern = dict_finder.find_all()
            # sort by length , long to short
            self.dict_pattern = sorted(self.dict_pattern, key=lambda x:len(x), reverse=True)
            for i in range(len(self.dict_pattern)):
                self.dict_pattern[i] = '(' + re.escape(self.dict_pattern[i]) + ')'
                # print(self.dict_pattern[i])
            self.res_label = self.re_find(self.dict_pattern) + self.res_label
            
            
            
            self.res_label = self.del_same(self.res_label)
            self.res_label = self.remove_overlamp(self.res_label)

        return self.res_label
    





if __name__=='__main__':
    finder = DEPARTMENT_Finder()
    finder.set_file(rf'.\data\file\file147.txt')
    
    print(finder.find() )