from LABEL_Finder.Finder import Finder
import re 

class DATE_Finder(Finder):
    def __init__(self):
        super().__init__()
        month = '(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        self.PATTERN = [ #DOCTOR
            r"(\d{1,2}[/\.]\d{1,2}[/\.]\d{2,5})" , 
            
            r'(\d{8})0000' , # FOR 199607180000
            
            r'DateOfBirth\n(\d{8})\n' ,  
            
            r'appendix (\d{4})' , 
            
            r'((?:\d{1,2} )?(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{2,5})' , # 1 January 1996
            
        ]
        self.res_label = []
        self.month_dict = {
            'January':1,
            'February':2,
            'March':3,
            'April':4,
            'May':5,
            'June':6,
            'July':7,
            'August':8,
            'September':9,
            'October':10,
            'November':11,
            'December':12,
        }
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        #self.res_label = self.del_same(self.res_label)
        
        
        #### patch for ----- 26th of April 2067
        pattern = r'(\d{1,2}th of (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{2,5})'
        p_res = self.re_find(pattern)
        
        # if len(p_res) > 0:
        #     print (p_res)
        self.res_label += p_res
        
        checked_label = []
        for lb in self.res_label:
            is_date = True
            rule_pattern = [ ##if match any rule , is_date = False
                #r'(?i)at ',     dont use 
                #r'(?i) at',     dont use 
                # r'(?i)on ',    dont use 
                # r'(?i) on',    dont use 
                
                r'\d{1,2}[:\.]\d{1,2}',
                r'of ' , 
                r'(?i)\dpm',
                r'(?i)\dam',
                r'\d{1,2}:\d{1,2}',
                r'at ?\d\d\d\d' , 
            ]
            w_size = 14
            for rule in rule_pattern:
                after_str = self.get_after_str(lb , w_size).split('\n')[0]
                before_str = self.get_before_str(lb , w_size).split('\n')[-1]
                # pass if it is close to other date xx.xx.xx
                if re.search(rule, after_str ) != None and re.search(r'\d{1,2}[\.]\d{1,2}[\.]\d{2,5}', after_str ) == None:
                    is_date = False
                    break
                if re.search(rule, before_str ) != None  and re.search(r'\d{1,2}[\.]\d{1,2}[\.]\d{2,5}', before_str ) == None:
                    is_date = False
                    break
            
            if is_date:
                checked_label.append(lb)
        self.res_label = checked_label
        
        return self.res_label
    
    def normalization(self , date_str):
        pattern = [                   #group index y  m  d
            [r'^(\d{1,2})/(\d{1,2})/(\d{2,5})$'     ,3, 2, 1 ]           ,
            [r'^(\d{1,2})\.(\d{1,2})\.(\d{2,5})$'   ,3, 2, 1 ]           ,
            [r'^(\d\d\d\d)(\d\d)(\d\d)$'            ,1, 2, 3 ]           ,
        ]
        for p in pattern:
            mat = re.search( p[0] , date_str)
            if mat!=None:
                y = int(mat.group(p[1]))
                m = int(mat.group(p[2]))
                d = int(mat.group(p[3]))
                if y<100:
                    y = 2000+y
                res = f'{y}-{m:02d}-{d:02d}'
                return res
        ##### type '2069' => 2069
        mat = re.search( r'^(\d\d\d\d)$' , date_str)
        if mat!=None:
            res = f'{mat.group(1)}'
            return res
        
        #### type '12th of September 2013' => 2013-09-12
        pat = r'^(\d{1,2})th of (January|February|March|April|May|June|July|August|September|October|November|December) (\d{2,5})$'
        mat = re.search( pat , date_str)
        if mat!=None:
            y = int(mat.group(3))
            m = self.month_dict[mat.group(2)]
            d = int(mat.group(1))
            res = f'{y}-{m:02d}-{d:02d}'
            return res
        
        #### type '12 September 2013' => 2013-09-12
        pat = r'^(\d{1,2}) (January|February|March|April|May|June|July|August|September|October|November|December) (\d{2,5})$'
        mat = re.search( pat , date_str)
        if mat!=None:
            y = int(mat.group(3))
            m = self.month_dict[mat.group(2)]
            d = int(mat.group(1))
            res = f'{y}-{m:02d}-{d:02d}'
            return res
        
        #### type 'September 2013' => 2013-09
        pat = r'^(January|February|March|April|May|June|July|August|September|October|November|December) (\d{2,5})$'
        mat = re.search( pat , date_str)
        if mat!=None:
            y = int(mat.group(2))
            m = self.month_dict[mat.group(1)]
            res = f'{y}-{m:02d}'
            return res
        
        
        
        
        return f'sorry date_str:{date_str} type  not match'
    
    def date_normalization(self , lb):
        res_lb = []
        for l in lb:
            res_lb.append( [*l , self.normalization(l[2])]  )            
        return res_lb
    
    
    
    def MD_change(self , lb):
        res_lb = []
    
    
    def patch_change_MMDD(self , labels):
        #cnt 1-12 times
        number_cnt = [0]*12
        for i , lb in enumerate(labels):
            #if i == 0:
            #    continue
            s,e,w,t = lb
            
            mat = re.search(r'(\d{2,4})-(\d{1,2})-(\d{1,2})' , t)
            if mat != None:
                if int(mat.group(1))<2000:
                    continue
                m = int(mat.group(2))
                d = int(mat.group(3))
                if m<=12:
                    number_cnt[m-1] += 1
                if d<=12:
                    number_cnt[d-1] += 1
            
        # start change 
        for lb in labels:
            s,e,w,t = lb
            mat = re.search(r'(\d{2,4})-(\d{1,2})-(\d{1,2})' , t)
            if mat != None:
                y = int(mat.group(1))
                m = int(mat.group(2))
                d = int(mat.group(3))
                if m>12:
                    lb[3] = f'{y}-{d:02d}-{m:02d}'
                    continue
                if m<=12 and d<=12 and number_cnt[m-1] < number_cnt[d-1]:
                    #print(f'change {t} to {y}-{d:02d}-{m:02d}')
                    lb[3] = f'{y}-{d:02d}-{m:02d}'
                    
        # print(number_cnt)
                    

        return labels
                
        
            
        



if __name__=='__main__':
    finder = DATE_Finder()
    finder.set_file(rf'.\data\file\file1809.txt')
    lb = finder.find()
    lb = finder.date_normalization(lb)
    
    for l in lb:
        print(l)
    
    
    
    
    finder.patch_change_MMDD(lb)
    print('--------- after patch    --------')
    for l in lb:
        print(l)
    
    