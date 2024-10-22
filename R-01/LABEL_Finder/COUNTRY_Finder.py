from LABEL_Finder.Finder import Finder
import re 

class COUNTRY_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #
            r'(Australia)',
            r'(Vietnam)',
            r'(South Africa)',
            r'\b(USA)\b',
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        
        self.res_label = self.del_same(self.res_label)
        
        new_label = []
        for lb in self.res_label:
            s,e,w = lb
            after = self.get_after_str(lb , 10)
            if '\n' in after:
                after = after[:after.index('\n')]
            if re.match(r'^ *\d{4}', after) == None:
                new_label.append(lb)
        self.res_label = new_label
        return self.res_label
    

if __name__=='__main__':
    finder = COUNTRY_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )