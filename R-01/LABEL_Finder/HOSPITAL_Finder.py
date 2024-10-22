from LABEL_Finder.Finder import Finder
import re 

class HOSPITAL_Finder(Finder):
    def __init__(self):
        super().__init__()
        
        self.PATTERN = [ #PATIENT
            # rf'Site_name: *([A-Z]+ HOSPITAL(?: & COMMUNITY HEALTH)?(?: CENTRE)?)', # no use
            #rf'Site_name: *([A-Z ]+ HOSPITAL(?:[A-Z\(\)\-])*)\n',
            
            
            ###### sel start
            r'Site_name: *(.{6,})\n',
            rf'Site: *([A-Z ]+ HOSPITAL(?:[A-Z ])*)\n',
            rf'\nSiteName\n(.+)\n', #0.9971
            rf"Location:.*?- *(?:OPERATIVE UNIT|OPERATIVEUNIT)?+[\- ]*([' &A-Z/\-\(\)]+) *\n" ,  #0.994
            ###### sel end

            
            # rf"((?:[A-Z']+ )+HOSPITAL(?: *\([A-Z]+\))?)" , 
            # rf"((?:[A-Z]+ )+HOSPITAL)" , 
            
            
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        self.res_label = self.del_same(self.res_label)
        
        
        res_add = []
        # V1
        # res = self.re_find(r"[- ]++([A-Z&\-']+(?: [A-Z&\-\(\)]+(?:'S)?)+\b\)?)")# v1 0.0.938
        
        
        # V2
        # res = self.re_find(r"[/\(\n\d\t- ]++([A-Z&\-']+(?: [A-Z&/\-\(]+(?:'S)?)+\b\)?)") # v2 0.9302
        # res += self.re_find(r"[/\(\n\d\t- ]++([A-Z&\-']+(?: [A-Z&/\-\(]+(?:'S)?)+\b)")
        # V2 end
        
        # V3
        res = self.re_find(r"[/\(\n\d\t- ]++([A-Z&\-']+(?: [A-Z&/\-]+(?:'S)?)+\b(?: ?\([A-Z]+\))?)")
        # V3 end        
        
        #print(res)
        for r in res:
            #print (r)
            
            if re.search(rf'SERVICE|HEALTH|HOSPITAL|CENTRE|CAMPUS', r[2]) != None:
                if (re.search(rf'\(', r[2])!= None) == (re.search(rf'\)', r[2])!= None):
                    #### M1
                    del_word = '&'
                    if  r[2][len(r[2])-len(del_word):len(r[2])]==del_word:
                        r[2] = r[2][0:len(r[2])-len(del_word)]
                        r[1] -= len(del_word)
                        
                    del_word = ' REPORTS' # this case is few
                    if  r[2][len(r[2])-len(del_word):len(r[2])]==del_word:
                        r[2] = r[2][0:len(r[2])-len(del_word)]
                        r[1] -= len(del_word)
                        
                    del_word = ' INSTITUTE' # this case is few
                    if  r[2][len(r[2])-len(del_word):len(r[2])]==del_word:
                        r[2] = r[2][0:len(r[2])-len(del_word)]
                        r[1] -= len(del_word)

                    
                    ##### the case of 'XXXX XXXX XXXX HOSPITAL HOSPITAL LABORATORIES'
                    # need del HOSPITAL LABORATORIES
                    del_word = ' HOSPITAL LABORATORIES'
                    if  r[2][len(r[2])-len(del_word):len(r[2])]==del_word:
                        r[2] = r[2][0:len(r[2])-len(del_word)]
                        r[1] -= len(del_word)
                        
                    ##### the case of 'XXXX XXXX XXXX HOSPITAL LABORATORIES'
                    # need del HOSPITAL LABORATORIES
                    del_word = ' LABORATORIES'
                    if  r[2][len(r[2])-len(del_word):len(r[2])]==del_word:
                        r[2] = r[2][0:len(r[2])-len(del_word)]
                        r[1] -= len(del_word)
                    
                    
                    if r not in self.res_label and r not in res_add:
                        res_add.append(r)
        self.res_label.extend(res_add)
        
        self.res_label = self.del_same(self.res_label)
        self.res_label = self.remove_overlamp(self.res_label)    
        return self.res_label
    





if __name__=='__main__':
    finder = HOSPITAL_Finder()
    finder.set_file(rf'.\data\file\file147.txt')
    
    print(finder.find() )