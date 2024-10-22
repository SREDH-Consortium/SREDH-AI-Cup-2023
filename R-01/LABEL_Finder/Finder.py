import re


class Finder:
    def __init__(self):
        self.file = ''
        
        
    def set_file(self, file_path):
        with open(file_path, 'r' , encoding='utf-8') as f:
            self.file = f.read()
            f.close()

    def set_file_str(self, file_str):
        self.file = file_str

    
    def re_find(self,pattern):
        res_array = []
        if type(pattern) == str:
            pattern = [pattern]
        for p in pattern:
            iter = re.finditer(p , self.file)
            for w in iter:
                g_count = w.lastindex
                for i in range(1,g_count+1):
                    res_array.append( [ w.span(i)[0], w.span(i)[1], w.group(i) ] )
        
        return res_array #format [[start,end,word],[start,end,word],...]
    def del_same(self, arr):
        res = []
        for a in arr:
            if a not in res:
                res.append(a)
        return res
    
    def res_filter(self,pattern , label_arr):
        res = []
        for lb in label_arr:
            s,e,w = lb
            if re.fullmatch(pattern, w)==None:
                res.append(lb)
        return res
    
    def is_overlap(self, lb1, lb2):
        s1,e1,w1 = lb1
        s2,e2,w2 = lb2
        if e1 >= s2 and s1 <= e2:
            return True
        return False
    
    def remove_overlamp(self, label_arr):
        # Check for overlap  ### if end1 >= start2 and start1 <= end2:
        res = []
        for lb in label_arr:
            ol = False
            for r in res:
                if self.is_overlap(lb,r):
                    ol = True
                    break
            if not ol:
                res.append(lb)
        return res
    
    def get_before_str(self , lb , size):
        s,e,w = lb
        if s-size < 0:
            return self.file[:s]
        return self.file[s-size:s]
    def get_after_str(self , lb , size):
        s,e,w = lb
        if e+size > len(self.file):
            return self.file[e:]
        return self.file[e:e+size]
    
    def get_label_line(self, lb):
        s,e,w = lb
        new_s = s
        new_e = e
        for i in range(s,-1,-1):
            if self.file[i] == '\n':
                break
            new_s -= 1
        for i in range(e,len(self.file)):
            if self.file[i] == '\n':
                break
            new_e += 1
        return [ new_s , new_e ,  self.file[new_s:new_e]]
    
    def len_filter(self , label_arr ,  min_len=None , max_len=None):
        res = []
        if min_len==None and max_len==None:
            return label_arr
        elif min_len==None:
            for lb in label_arr:
                s,e,w = lb
                if len(w) <= max_len:
                    res.append(lb)
        elif max_len==None:
            for lb in label_arr:
                s,e,w = lb
                if len(w) >= min_len:
                    res.append(lb)
        else:
            for lb in label_arr:
                s,e,w = lb
                if len(w) >= min_len and len(w) <= max_len:
                    res.append(lb)
        return res

if __name__=='__main__':
    finder = Finder()
    finder.set_file(rf'.\data\answer.txt')
    
    p = ["(IDNUM)\t(20)","(IDNUM)\t(20)","(IDNUM)\t(20)"]
    print(finder.re_find(p) )
        
        
        