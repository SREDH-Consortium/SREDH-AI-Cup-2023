from LABEL_Finder.Finder import Finder
import re 

class ZIP_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #DOCTOR
            
        ]
        self.res_label = []
        
        
    def find(self):
        #self.res_label = self.re_find(self.PATTERN)
        #self.res_label = self.del_same(self.res_label)
        
        #pattern = r'Lab No:.*\n(.+)\n' ### V1
        # case MOUNT BARKER  VIC2441 ==> 2441
        pattern = r'Lab ?No:(?:.*\n){1,3}?.+[ A-Z](\d\d\d\d)\n'
        
        self.res_label = self.re_find(pattern)
        self.res_label = self.res_filter(r'.{,3}',self.res_label)
        
        return self.res_label
    

if __name__=='__main__':
    finder = ZIP_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )