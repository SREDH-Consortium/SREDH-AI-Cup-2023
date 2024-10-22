from LABEL_Finder.Finder import Finder
import re 

class DOCTOR_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            
            
            
            
            
            
            r'(?i)[ \n]DR +([A-Za-z\-]* ?[A-Za-z\-]* ?[A-Za-z\-]*)[\n)]', # 100% correct 
            rf'(?i:Reported ?by ?Dr *)([A-Z][A-Za-z\-]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?)' , #~100% correct
            rf'\(([A-Z][A-Z])/ ?ec' , 
            
            
            rf'[Dd][Rr]\.? ([A-Z][A-Za-z\-\.]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?)' ,# this will get some wrong , but correct much 
            
            rf"Prof\.? ([A-Z][A-Za-z\-]*(?: [A-Z]['A-Za-z\-]*)?(?: [A-Z]['A-Za-z\-]*)?)" , # this will get some wrong , but correct much 
            
            
            #rf'\(TO: ?([A-Z][A-Z]);? ?([A-Z][A-Z])?/'        , # this cant apply to filexxxx.txt , but xxxx.txt is must
            # add for first phase vaildation
            # case Reported by W Vasso
            rf'(?i:Reported ?by )([A-Z][A-Za-z\-]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?)',
            
            rf'(?i)\n ?MICROSCOPIC *:? *\(Reported by Dr ([A-Z][A-Za-z\-]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?) *\)',
            rf'(?i)\n ?MICROSCOPIC *:? *\(Reported by ([A-Z][A-Za-z\-]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?) *\)',
            rf'(?i)\n ?MICROSCOPIC *:? *\(Reported ([A-Z][A-Za-z\-]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?) *\)',
            rf'(?i)\n ?MICROSCOPIC *:? *\(([A-Z][A-Za-z\-]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?) *\)',
            
            # PRO Somes
            rf'PRO ([A-Z][A-Za-z\-]*(?: [A-Z][A-Za-z\-]*)?(?: [A-Z][A-Za-z\-]*)?)\b',
            
            
        ]
        self.res_label = []
        
        
    def find(self,file_name):
        # # get key key line 'MICROSCOPIC (Reported by Dr L Litchard):'
        # key_lb = self.re_find(rf'\n ?MICROSCOPIC')
        # key_line = []
        # for lb in key_lb:
        #     key_line.append(self.get_label_line(lb)[2])
        
        
        
        
        
        self.res_label = self.re_find(self.PATTERN)

        if 'file' not in file_name: # filexxxx.txt
            self.res_label += self.re_find(rf'\(TO: *([A-Z][A-Z])[;:]? *([A-Z][A-Z])?/')
            self.res_label += self.re_find(rf'\(([A-Z][A-Z])/')

        self.res_label = self.del_same(self.res_label)
        
        self.res_label = self.remove_overlamp(self.res_label)
        
        # len must > 1
        new_res_label = []
        for i in self.res_label:
            if len(i[2]) > 1:
                new_res_label.append(i)        
        self.res_label = new_res_label
        # filter 'Dr'
        self.res_label = self.res_filter(r'Dr' , self.res_label)        
        self.res_label = self.res_filter(r'^(?:HM|PG|RR|FP|ED|JL|CA|MS)$' , self.res_label)        
        
        
        return self.res_label
    


if __name__=='__main__':
    finder = DOCTOR_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )