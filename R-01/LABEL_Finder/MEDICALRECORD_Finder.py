from LABEL_Finder.Finder import Finder
import re 

class MEDICALRECORD_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            rf'(?i)MRN no: *(\w+)\n?'        ,
            r'(\d{6,8}\.[A-Z][A-Z][A-Z])'      ,
            rf'\nMRN\n(\d+)\n'        ,
            rf'\nMRN: ?(\d+(?:\.[A-Z][A-Z][A-Z])?)'        ,
            
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        #self.res_label = self.del_same(self.res_label)
        
        return self.res_label
    



if __name__=='__main__':
    finder = MEDICALRECORD_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )