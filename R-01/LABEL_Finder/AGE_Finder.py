from LABEL_Finder.Finder import Finder
import re 

class AGE_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            r'(?i)(\d{2,3}) *years *old',
            r'(?i)(\d{2,3}) *year *old',
            r'(?i)(\d{2,3}) *yrs *old',
            r'(?i)(\d{2,3}) *yr *old',
            r'(?i)(\d{2,3}) *year-old',
            r'(?i)(\d{2,3})-year-old',
            r'(?i)(\d{2,3}) *year\b',
            r'(?i)(\d{2,3}) *old\b',
            r'(?i)(\d{2,3}) *yo\b',
            r'(?i)(\d{2,3}) *y\.o\.',
            r'(?i)\n(\d{2,3}) ?F\b', ##(?!\.)
            r'(?i)\n(\d{2,3}) ?M\b', ##(?!\.)
            r'(?i)\n(\d{2,3})yr\b',
            r'(?i)\bage *(\d{2,3})\b',
            
            # for file29331 56 female
            r'(?i)(\d{2,3}) *female',
            r'(?i)(\d{2,3}) *male',
            
            # 
            r'cancer at (\d{2,3})s?\b',
            r'cancer in (\d{2,3})s?\b',
            # r'cancer (\d{2,3})\b',
            
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        self.res_label = self.del_same(self.res_label)
        
        self.res_label = self.remove_overlamp(self.res_label) # just add
        
        return self.res_label
    

if __name__=='__main__':
    finder = AGE_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )