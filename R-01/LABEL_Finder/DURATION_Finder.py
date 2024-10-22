from LABEL_Finder.Finder import Finder
import re 

class DURATION_Finder(Finder):
    def __init__(self):
        super().__init__()
        #['zero', 'one' , 'two' , 'three' , 'four' , 'five' , 'six' , 'seven' , 'eight' , 'nine' , 'ten' , 'eleven' , 'twelve' , 'thirteen' , 'fourteen' , 'fifteen' , 'sixteen' , 'seventeen' , 'eightteen' , 'nineteen' , 'twenty' ]
        self.ENGLISH_NUM = {
            'zero' : '0',
            'one' : '1',
            'two' : '2',
            'three' : '3',
            'four' : '4',
            'five' : '5',
            'six' : '6',
            'seven' : '7',
            'eight' : '8',
            'nine' : '9',
            'ten' : '10',
            'eleven' : '11',
            'twelve' : '12',
            'thirteen' : '13',
            'fourteen' : '14',
            'fifteen' : '15',
            'sixteen' : '16',
            'seventeen' : '17',
            'eightteen' : '18',
            'nineteen' : '19',
            'twenty' : '20',
        }
        
        
        self.PATTERN = [ #DOCTOR
            ### '(\d{1,3}(?:-\d{1,3})?)'                        
                        
            r'(?i)(\d{1,3}(?:-\d{1,3})? *years)\b(?! old)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *year)\b(?! old)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *yrs)\b(?! old)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *yr\b)\b(?! old)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *weeks)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *week)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *wks)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *months)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *month)',
            
            r'(?i)(\d{1,3}(?:-\d{1,3})? *day)',
            r'(?i)(\d{1,3}(?:-\d{1,3})? *days)',
            
            # one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
            # r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) *years',
            # (?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty)
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +years)(?! old)',
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +yrs)(?! old)',
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +yr\b)(?! old)',
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +weeks)',
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +week)',
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +wks)',
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +months)',
            r'(?i)((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eightteen|nineteen|twenty) +month)',
            
            
        ]
        self.res_label = []
        
        
    def find(self):
        self.res_label = self.re_find(self.PATTERN)
        self.res_label = self.del_same(self.res_label)
        self.res_label = self.remove_overlamp(self.res_label)
        
        
        ## patch for FIRST_PHASE_VALIDATION 887.txt 4-5 month 4.5month
        # for lb in self.res_label:
        #     # get before str see if \d*- before it
        #     before_str = self.get_before_str(lb)
        #     if before_str!=None:
        #         pattern = r'(\d{1,3})-$'
        #         mat = re.search(pattern , before_str)
        #         if mat!=None:
        #             lb[0] = mat.group(1) + ' - ' + lb[0]
        #             lb[2] = self.normalization(lb[0])
        #             break
        
        return self.res_label
    
    def normalization(self , date_str):
        pattern = r'(\d{1,3}(?:-\d{1,3})?) *(?:years|year|yrs|yr)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1).replace('-' , '.')
            return f'P{num}Y'
        
        pattern = r'(\d{1,3}(?:-\d{1,3})?) *(?:weeks|week|wks)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1).replace('-' , '.')
            return f'P{num}W'
        
        pattern = r'(\d{1,3}(?:-\d{1,3})?) *(?:months|month)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1).replace('-' , '.')
            return f'P{num}M'
        
        pattern = r'(\d{1,3}(?:-\d{1,3})?) *(?:days|day)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1).replace('-' , '.')
            return f'P{num}D'
        
        # engilsh num
        pattern = r'(\w+) *(?:years|yrs|yr)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1)
            if num in self.ENGLISH_NUM:
                num = self.ENGLISH_NUM[num]
            return f'P{num}Y'
        
        pattern = r'(\w+) *(?:weeks|week|wks)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1)
            if num in self.ENGLISH_NUM:
                num = self.ENGLISH_NUM[num]
            return f'P{num}W'
        
        pattern = r'(\w+) *(?:months|month)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1)
            if num in self.ENGLISH_NUM:
                num = self.ENGLISH_NUM[num]
            return f'P{num}M'
        
        pattern = r'(\w+) *(?:days|day)'
        mat = re.search(pattern , date_str)
        if mat!=None:
            num = mat.group(1)
            if num in self.ENGLISH_NUM:
                num = self.ENGLISH_NUM[num]
            return f'P{num}M'
        
    
    
    def duration_normalization(self , lb):
        res_lb = []
        for l in lb:
            res_lb.append( [ *l , self.normalization(l[2]) ] )
        return res_lb

if __name__=='__main__':
    finder = DURATION_Finder()
    finder.set_file(rf'.\data\file\10.txt')
    
    
    print(finder.find() )