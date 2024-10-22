from LABEL_Finder.Finder import Finder
import re 

class PATIENT_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #PATIENT
            #rf'\n([A-Z][A-Za-z]*(?:[, ]+[A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?) ?\n' ,
            
            rf"(?<![Dd][Rr] )(\b[A-Z][A-Za-z]*(?: ?, [A-Z][A-Za-z\-']*,?)(?: [A-Z][A-Za-z\-']*)?(?: [A-Z][A-Za-z\-']*)?) *\n" , #each file only have one 
            
            rf'\nFirstName\n([A-Z][A-Za-z]*)\n'    ,
            rf'\nMiddleName\n([A-Z][A-Za-z\-]*)\n' ,
            rf'\nLastName\n([A-Z][A-Za-z\-]*)\n' ,
            
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN[0])
        if len(self.res_label)>0:
            self.res_label = [self.res_label[0]]
        # if len > 1        
        self.res_label += self.re_find(self.PATTERN[1:len(self.PATTERN)])
            
        
        self.res_label = self.del_same(self.res_label)    
        return self.res_label
    





if __name__=='__main__':
    finder = PATIENT_finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    print(finder.find() )