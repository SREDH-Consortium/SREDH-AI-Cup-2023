from LABEL_Finder.Finder import Finder
import re 

class SET_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(r'\b(once)\b')
        self.res_label += self.re_find(r'\b(twice)\b')
        
        self.res_label = self.del_same(self.res_label)
        return self.res_label
    
    
    def normalization(self , lb):
        if 'once' in lb:
            return 'R1'
        elif 'twice' in lb:
            return 'R2'
        return 'R0'
    
    
    def set_normalization(self , lb):
        res_lb = []
        for l in lb:
            res_lb.append( [*l , self.normalization(l[2])]  )            
        return res_lb

if __name__=='__main__':
    finder = SET_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )