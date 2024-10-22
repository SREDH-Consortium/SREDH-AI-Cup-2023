from LABEL_Finder.Finder import Finder
import re 

class URL_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            
        ]
        self.res_label = []
        
        
    def find(self):
        pattern = r'(http[s]?:/[^\n^"^\(^\)]+)'
        
        self.res_label = self.re_find(pattern)
        self.res_label = self.del_same(self.res_label)
        self.res_label = self.remove_overlamp(self.res_label)        
        return self.res_label
    

if __name__=='__main__':
    finder = URL_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )