class Finder_DICT:
    def __init__(self , dict_path) -> None:
        
        self.dict = ''
        with open(dict_path, 'r' , encoding='utf-8') as f:
            self.dict = f.read()
        self.dict = self.dict.split('\n')
        # delete empty string
        self.dict = list(filter(None, self.dict))

        self.file = ''
    
    def set_file_path(self , path):# path
        with open(path, 'r' , encoding='utf-8') as f:
            self.file = f.read()
    def set_file(self , file):# str
        self.file = file
    
    def dict_len_filter(self , min_len , max_len=None):
        new_dict = []
        for word in self.dict:
            if max_len == None:
                if len(word) >= min_len:
                    new_dict.append(word)
            else:
                if len(word) >= min_len and len(word) <= max_len:
                    new_dict.append(word)
        self.dict = new_dict
    
    def find_all(self):
        res = []
        for word in self.dict:
            if word in self.file:
                res.append(word)
        return res
    
    

    
if __name__ == '__main__':
    dict_finder = Finder_DICT(rf'dictionary\CITY_dict.txt')
    
    dict_finder.set_file_path(rf'data\file\10.txt')
    
    res = dict_finder.find_all()
    
    
    print(res)
    
    
    
        
    