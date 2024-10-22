from LABEL_Finder.Finder import Finder
import re 

class STREET_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            
        ]
        self.res_label = []
        
        
    def find(self):
        #self.res_label = self.re_find(self.PATTERN)
        #self.res_label = self.del_same(self.res_label)
        
        #pattern = r'Lab No:.*\n(.+)\n' ### V1
        #pattern = r'Lab ?No:.*\n([A-Z][a-z\.]*(?: [A-Zo][a-z\.]*)*)\b' # 'o' is for of   ### V2 # Xxxxx of Xxxxx
        pattern = r'Lab ?No:.*\n([A-Z][A-Za-z\.]*(?: +[A-Zo][a-z\.]*)*)\b'### V3  # 'o' is for of  # Xxxxx of Xxxxx
        self.res_label = self.re_find(pattern)
        self.res_label = self.res_filter(r'.{,3}',self.res_label)
        
        return self.res_label
    

if __name__=='__main__':
    finder = STREET_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )