from LABEL_Finder.Finder import Finder
from LABEL_Finder.STATE_Finder import STATE_Finder
from LABEL_Finder.ZIP_Finder import ZIP_Finder
from LABEL_Finder.Finder_DICT import Finder_DICT
import re 

class CITY_Finder(Finder):
    def __init__(self , dict_path = rf'dictionary\CITY_force_dict.txt'):
        super().__init__()
        self.dict_path = dict_path
        self.dict_pattern = []
        self.PATTERN = [ #CITY_Finder
        ]
        self.res_label = []
        
    def find(self):
        ###### just test dont have any help#####
        # # by dictonary first 
        # dict_finder = Finder_DICT(self.dict_path)
        # dict_finder.set_file(self.file)
        # self.dict_pattern = dict_finder.find_all()
        # # sort by length , long to short
        # self.dict_pattern = sorted(self.dict_pattern, key=lambda x:len(x), reverse=True)
        # for i in range(len(self.dict_pattern)):
        #     self.dict_pattern[i] = r'\b(' + self.dict_pattern[i] + r')\b'
        # self.res_label = self.re_find(self.dict_pattern)
        # # label index must >1000
        # checked = []
        # for lb in self.res_label:
        #     if lb[0] > 1000:
        #         checked.append(lb)
        # self.res_label = checked
        
        state_finder = STATE_Finder()
        state_finder.set_file_str(self.file)
        st_res = state_finder.find()
        
        for st in st_res: #this for  CITY	0.9953052	0.9030884	0.946957	939
            s,e,w = st
            line_start, line_end, line = self.get_label_line(st)
            
            lb_s = line_start
            lb_e = s
            
            city = self.file[lb_s:lb_e]
            mat = re.search(r'([A-Z]+(?: *[A-Z\-]+)*)', city)
            if mat == None:
                continue
            city = mat.group(1)
            lb = self.re_find( '(' + city + ')' )[0]
            self.res_label.append(lb)
        
        # 11/25 add some case dont have state in that line 
        # will use zip to find city
        if len(self.res_label) == 0:
            zip_finder = ZIP_Finder()
            zip_finder.set_file_str(self.file)
            zip_res = zip_finder.find()
            for zi in zip_res: #this for  CITY	0.9953052	0.9030884	0.946957	939
                s,e,w = zi
                line_start, line_end, line = self.get_label_line(zi)
                
                lb_s = line_start
                lb_e = s
                
                city = self.file[lb_s:lb_e]
                mat = re.search(r'([A-Z]+(?: *[A-Z\-]+)*)', city)
                if mat == None:
                    continue
                city = mat.group(1)
                lb = self.re_find( '(' + city + ')' )[0]
                self.res_label.append(lb)
        
        
                
        return self.res_label
    



if __name__=='__main__':
    finder = CITY_Finder()
    
    finder.set_file(rf'.\data\file\151.txt')
    
    res = finder.find()
    
    print(res)
        
    

    #finder.set_file(rf'.\data\file\10.txt')
    
    
    #print(finder.find() )