from LABEL_Finder.Finder import Finder
import re 

class TIME_Finder(Finder):
    def __init__(self):
        super().__init__()
        self.PATTERN_time = r'(?i)(\d{1,2}[:\.]\d{1,2} ?(?:am|pm|hr|hrs)?)'
        self.PATTERN_date = r'\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4}'
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
        self.PATTERN = [ #
            #######################
            ##### DATE FIIRST #####            
            #######################
                        
            #case 2679-02-10 00:00:00
            r'(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d)'               ,
            
            ### case 30/12/2013 12:00 AM
            ### case 30.12.2013 at 12.00 am
            ### case 30/12/2013 @ 12:00AM
            ### case 30/12/2013at 12:00 AM
            ### case 30/12/2013 on12:00 AM
            ### case 30/12/2013 on 5.00 AM
            r'(?i)(\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4} *(?:at|and|on|@)? *\d{1,2}[:\.]\d{1,2}(?: ?(?:am|pm|hrs|hr))?)'         ,
            
            #case 02.04.15 at 1550hrs
            r'(?i)(\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4} *(?:on the|on|at|and|,|@)? *\d{4}(?: ?(?:am|pm|hrs|hr))?)'         ,
            #case 28.07.20 12pm
            r'(?i)(\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4} +(?:on the|on|at|and|,|@)? *\d{1,2}(?: ?(?:am|pm|hrs|hr))?)'         ,
            #######################
            ##### TIME FIRST #####
            #######################
            
            # case 12:15 on 28.3.13
            # case 3:26pm on 7/7/21
            ### case 12:00 on 28.3.13
            ### case 12.00 on 28/3/2013
            ### case 12.00pm on 28/3/2013
            ### case 12.00 pm on 28/3/2013
            ### case 12:00 at 28.3.13
            ### case 12.00 at 28/3/2013
            ### case 12.00pm at 28/3/2013
            ### case 12.00 pm at 28/3/2013
            ### case 9.50am and 14/3/13
            r'(\d{1,2}[:\.]\d{1,2} ?(?:am|pm|hr|hrs)? *(?:on the|on|at|and|,|@)? *\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4})',
            
            #case  1230 on 20/09/16 
            r'(\d{4} ?(?:am|pm|hr|hrs)? *(?:on|at|and) *\d{1,2}[/\.]\d{1,2}[/\.]\d{2,4})',
            
            
            
            #case '10:10am on the 17th of September 2013'
            r'(\d{1,2}:\d{1,2}(?:am|pm) on the \d\d(?:th|st|nd|rd) of (?:January|February|March|April|May|June|July|August|September|October|November|December) \d\d\d\d)'               ,
            # case 1500pm on the 28th of October 2013
            r'(\d{4} ?(?:am|pm|hours) on the \d\d(?:th|st|nd|rd) of (?:January|February|March|April|May|June|July|August|September|October|November|December) \d\d\d\d)'               ,
            # 2pm on 4/11/13
            r'(\d{1,2}(?:am|pm|hours) on \d{1,2}[/\.]\d{1,2}[/\.]\d{2,4})'               ,
            
            #case 'at 10:10am on the 17th of September 2013'

            
            #case 5:45 pm , cant use , 
            #r'(?i)(\d{1,2}[:\.]\d{1,2} *(?:am|pm|hr|hrs)?)'   ,
            
            ## those is too few , i will not solve ###paybe solve in at program end 
            #5th of November 2013 at 10:15am
            r'(\d{1,2}(?:th|st|nd|rd) of (?:January|February|March|April|May|June|July|August|September|October|November|December) \d\d\d\d at \d{1,2}:\d{1,2}(?:am|pm))'               ,
            #12:55 p.m. on the 12th of September 2013
            r'(\d{1,2}:\d{1,2} (?:am|pm|a\.m\.|p\.m\.) on the \d\d(?:th|st|nd|rd) of (?:January|February|March|April|May|June|July|August|September|October|November|December) \d\d\d\d)'               ,
            #12th of December 2013 at 1740
            r'(\d\d(?:th|st|nd|rd) of (?:January|February|March|April|May|June|July|August|September|October|November|December) \d\d\d\d at \d{4})'               ,

            #######################
            ##### TIME only   #####           
            #######################
            # not use
            # r'((?:at )?\d{1,2}[:\.]\d{1,2} ?(?:am|pm|hr|hrs))',
            
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        self.res_label = self.del_same(self.res_label)
        
        self.res_label = self.remove_overlamp(self.res_label)
        return self.res_label
    
    def normalization(self , date_str):
        
        pattern = r'(\d\d\d\d-\d\d-\d\d) (\d\d:\d\d:\d\d)'
        mat = re.search(pattern, date_str)
        if mat != None:
            date = mat.group(1)
            time = mat.group(2)
            res = f'{date}T{time}'
            return res

        pattern = r'(?i)(\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4}) *(?:at|and|on|@)? *(\d{1,2})[:\.](\d{1,2})(?: ?(?:am|pm|hrs|hr))?'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(3))
            m = int(mat.group(2))
            d = int(mat.group(1))
            hh = int(mat.group(4))
            mm = int(mat.group(5))
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        #case 02.04.15 at 1550hrs
        pattern = r'(?i)(\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4}) *(?:on the|on|at|and|,|@)? *(\d{4})(?: ?(?:am|pm|hrs|hr))?'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(3))
            m = int(mat.group(2))
            d = int(mat.group(1))
            hh = int(mat.group(4)[0:2])
            mm = int(mat.group(4)[2:4])
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        #case 28.07.20 12pm
        pattern = r'(?i)(\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4}) +(?:on the|on|at|and|,|@)? *(\d{1,2})(?: ?(?:am|pm|hrs|hr))?'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(3))
            m = int(mat.group(2))
            d = int(mat.group(1))
            hh = int(mat.group(4))
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:00'
            return res
        
        #######################
        ##### TIME FIRST #####
        #######################
        #case 12.00 pm at 28/3/2013
        pattern =  r'(\d{1,2})[:\.](\d{1,2}) ?(?:am|pm|hr|hrs)? *(?:on the|on|at|and|,|@)? *(\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4})' 
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(5))
            m = int(mat.group(4))
            d = int(mat.group(3))
            hh = int(mat.group(1))
            mm = int(mat.group(2))
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        #case  1230 on 20/09/16 
        pattern = r'(\d{4}) ?(?:am|pm|hr|hrs)? *(?:on|at|and) *(\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4})'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(4))
            m = int(mat.group(3))
            d = int(mat.group(2))
            hh = int(mat.group(1)[0:2])
            mm = int(mat.group(1)[2:4])
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        
        #case '10:10am on the 17th of September 2013'
        pattern = r'(\d{1,2}):(\d{1,2})(?:am|pm) on the (\d\d)(?:th|st|nd|rd) of (January|February|March|April|May|June|July|August|September|October|November|December) (\d\d\d\d)'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(5))
            m = self.month_dict[mat.group(4)]
            d = int(mat.group(3))
            hh = int(mat.group(1))
            mm = int(mat.group(2))
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        # case 1500pm on the 28th of October 2013
        pattern = r'(\d{4}) ?(?:am|pm|hours) on the (\d\d)(?:th|st|nd|rd) of (January|February|March|April|May|June|July|August|September|October|November|December) (\d\d\d\d)'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(4))
            m = self.month_dict[mat.group(3)]
            d = int(mat.group(2))
            hh = int(mat.group(1)[0:2])
            mm = int(mat.group(1)[2:4])
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        # 2pm on 4/11/13
        pattern = r'(\d{1,2})(?:am|pm|hours) on (\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4})'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(4))
            m = int(mat.group(3))
            d = int(mat.group(2))
            hh = int(mat.group(1))
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:00'
            return res
        
        ########################### very few case ##########################
        
        #5th of November 2013 at 10:15am
        pattern = r'(\d{1,2})(?:th|st|nd|rd) of (January|February|March|April|May|June|July|August|September|October|November|December) (\d\d\d\d) at (\d{1,2}):(\d{1,2})(?:am|pm)'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(3))
            m = self.month_dict[mat.group(2)]
            d = int(mat.group(1))
            hh = int(mat.group(4))
            mm = int(mat.group(5))
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        #12:55 p.m. on the 12th of September 2013
        pattern = r'(\d{1,2}):(\d{1,2}) (?:am|pm|a\.m\.|p\.m\.) on the (\d\d)(?:th|st|nd|rd) of (January|February|March|April|May|June|July|August|September|October|November|December) (\d\d\d\d)'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(5))
            m = self.month_dict[mat.group(4)]
            d = int(mat.group(3))
            hh = int(mat.group(1))
            mm = int(mat.group(2))
            if y<100:
                y+=2000
            if hh<12 and 'pm' in date_str.lower():
                hh+=12
            elif hh<12 and 'p.m.' in date_str.lower():
                hh+=12
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        #12th of December 2013 at 1740
        pattern = r'(\d\d)(?:th|st|nd|rd) of (January|February|March|April|May|June|July|August|September|October|November|December) (\d\d\d\d) at (\d{4})'
        mat = re.search(pattern, date_str)
        if mat != None:
            y = int(mat.group(3))
            m = self.month_dict[mat.group(2)]
            d = int(mat.group(1))
            hh = int(mat.group(4)[0:2])
            mm = int(mat.group(4)[2:4])
            if y<100:
                y+=2000
            res = f'{y}-{m:02d}-{d:02d}T{hh:02d}:{mm:02d}'
            return res
        
        return f'sorry time_str:({date_str}) type  not match'
    
    
    
    def time_normalization(self , lb):
        res_lb = []
        for l in lb:
            res_lb.append( [ *l , self.normalization(l[2]) ] )
        return res_lb
    



if __name__=='__main__':
    finder = TIME_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    print(finder.find() )