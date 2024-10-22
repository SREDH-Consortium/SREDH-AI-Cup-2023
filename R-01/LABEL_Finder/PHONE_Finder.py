from LABEL_Finder.Finder import Finder
import re 

class PHONE_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            r'\((\d\d\d\d \d\d\d\d)\)',
            r'(?i)Phone: *(\d\d\d\d \d\d\d\d)',
            r'(?i)Tel: *(\d\d\d\d \d\d\d\d)', # does not exit in data
            
            # match (83829154)
            r'\((\d\d\d\d\d\d\d\d)\)',
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        self.res_label = self.del_same(self.res_label)
        
        return self.res_label
    

if __name__=='__main__':
    finder = PHONE_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )