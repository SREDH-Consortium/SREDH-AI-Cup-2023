from LABEL_Finder.Finder import Finder
import re 

### all state 

# Australian Capital Territory    32
# Northern Territory              13
# Tasmania                        33
# Queensland                      32
# New South Wales                 32
# Western Australia               28
# South Australia                 27
# Victoria                        23

# TAS                             32
# NT                              32
# SA                              28
# WA                              28
# QLD                             26
# VIC                             23
# NSW                              3
# ACT                             36



class STATE_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        ''' 
        self.PATTERN = [ #STATE
            r'\b(Australian Capital Territory)\b' ,
            r'\b(Northern Territory)\b'           ,
            r'\b(Tasmania)\b'                     ,
            r'\b(Queensland)\b'                   ,
            r'\b(New South Wales)\b'              ,
            r'\b(Western Australia)\b'            ,
            r'\b(South Australia)\b'              ,
            r'\b(Victoria)\b'                     ,

            r'\b(TAS)\b'                          ,
            r'\b(NT)\b'                           ,
            r'\b(SA)\b'                           ,
            r'\b(WA)\b'                           ,
            r'\b(QLD)\b'                          ,
            r'\b(VIC)\b'                          ,
            r'\b(NSW)\b'                          ,
            r'\b(ACT)\b'                          ,
        ]
        '''
        self.KEYWORD = [
            'Australian Capital Territory' ,
            'Northern Territory'           ,
            'Tasmania'                     ,
            'Queensland'                   ,
            'New South Wales'              ,
            'Western Australia'            ,
            'South Australia'              ,
            'Victoria'                     ,
            'TAS'                          ,
            'NT'                           ,
            'SA'                           ,
            'WA'                           ,
            'QLD'                          ,
            'VIC'                          ,
            'NSW'                          ,
            'ACT'                          ,
        ]
        self.PATTERN = []
        self.res_label = []
        for i in range(len(self.KEYWORD)):
            if len(self.KEYWORD[i])<5:
                # self.PATTERN[i] = r'\b('+self.PATTERN[i]+r')[ \d]*\n'#V1
                self.PATTERN.append( r'('+self.KEYWORD[i]+r') *\d\d\d\d' )
            else:
                # self.PATTERN[i] = r'('+self.PATTERN[i]+r')[ \d]*\n'#V1
                self.PATTERN.append( r'('+self.KEYWORD[i]+r') *\d\d\d\d' )
                
        

        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        self.res_label = self.del_same(self.res_label)
        self.res_label = self.remove_overlamp(self.res_label)
        
        # pattern = r'Lab ?No:.*\n([A-Z][A-Za-z\.]*(?: +[A-Zo][a-z\.]*)*)\b'### V3  # 'o' is for of  # Xxxxx of Xxxxx
        
        
        # patch for few case  "PUBLIC HOSPITAL ACT" , ACT is not state
        checked_label = []
        for lb in self.res_label:
            s,e,w = lb
            before = self.get_before_str(lb, 10)
            if re.search(r'HOSPITAL', before) == None:
                checked_label.append(lb)
        self.res_label = checked_label
        
        return self.res_label
    

if __name__=='__main__':
    finder = STATE_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )