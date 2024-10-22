from LABEL_Finder.Finder import Finder
import re 
class IDNUM_Finder(Finder):
    def __init__(self):
        super().__init__()
        self.PATTERN = [ #IDNUM
            rf'(?i)lab *no:? *(\w+),?(\w+)?'        ,
            rf'(?i)episode *no:? *(\w+),?(\w+)?'    ,
            rf'(?i)SPR *no:? *(\w+),?(\w+)?'        ,
            r'\A(?:.*+\n){1,9}?(\w{7,8}),?(\w{7,8})?\n' , #maybe it will have some problem , it need to del [A-Za-z]+
#
            r'\nSPRID\n(\w{7,11})\n'                ,
            r'Pathology Report(\w{7,11})\b (?!No:)'                   ,
            r'Pathology Report(\w{7,11})\b,(\w{7,11})'                   ,
            
            r'Laboratory Number: (\w{7,11})\b,(\w{7,11})'                   ,
            
            r'report (\w{7,11})\b'                              ,
            
            #find 22W23817,22W23817 
            r'[\n ](?P<id>\w{8,11}), ?(?P=id)'                   , # maybe it will same as patt 0,1,2 
            #r'[ \n](?P<id>\w{7,11}), ?(?P=id) '
            
            
            # 11/24 add
            r'SPRText\n(\w{7,11})\b',
            r'SPRText\n\w{7,11}\((\w{7,11})\)',
            # Pathology number 818540-31TH
            r'Pathology number ([A-Z\d\-]{7,12})\b',
            
            
            r'\(([A-Z\d]{7,11})\)',
            
            
        
        ]
        self.res_label = []

    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        self.res_label = self.del_same(self.res_label)
        ### patch
        
        #del word or number # r'[A-Za-z]+'
        self.res_label = self.res_filter(rf'[A-Za-z]+', self.res_label)
        self.res_label = self.res_filter(rf'[0-9]+', self.res_label)
        
        
        #add 
        new_label = []
        for lb in self.res_label:
            s,e,w = lb
            if len(w) < 6:
                continue
                        
            res = self.re_find(rf'\b({w})\b')
            #res += self.re_find(rf'Report({w})\b')
            for r in res:
                if r not in self.res_label and r not in new_label:
                    new_label.append(r)

        #if len(new_label) > 0:
        #    print('[IDNUM_Finder]:add:' , new_label)
            
        self.res_label.extend(new_label)
            
        
        
        
        return self.res_label
        
        
        
        
if __name__=='__main__':
    finder = IDNUM_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    print(finder.find() )
        
        
        
        
        
