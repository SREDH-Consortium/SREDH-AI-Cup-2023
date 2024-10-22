# read ORGANIZATION_dict_v2 
# delete key word

DELETE_WORDS = [
    'Co'            ,
    'Ltd'           ,
    'Inc'           ,
    'Corp'          ,
    'Corporation'   ,
    'Company'       ,
    'Group'         ,

]
DELETE_WORDS2  = [
    'Services'    ,
    'Service'     ,
    'USA'         ,
]

with open('dictionary/ORGANIZATION_dict_v2.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    res_line = []
    # delete dot if line end is '.'
    for line in lines:
        if ',' in line:
            line = line.split(',')[0]
        line = line.strip()
        while(line!='' and  line[-1] == '.'):
            line = line[:-1]
            line = line.strip()
        
        sp = line.split(' ')
        for word in DELETE_WORDS:
            if word in sp:
                sp.remove(word)
                break
        for word in DELETE_WORDS2:
            if word in sp:
                sp.remove(word)
                # break
        while sp[-1] == '&':
            sp.pop()
        line = ' '.join(sp)
        res_line.append(line)
    ### add line " & " -> " and "
    new_res_line = [] # " & " -> " and "
    for line in res_line:
        if ' & ' in line:
            line = line.replace(' & ', ' and ')
            new_res_line.append(line)
    res_line += new_res_line

    ### delete "'" in the end of line
    new_res_line = []
    for line in res_line:
        if line!= '' and line[-1] == "'":
            line = line[:-1]
        new_res_line.append(line)
    res_line = new_res_line
    
    

    with open('dictionary/ORGANIZATION_dict_v3.txt', 'w', encoding='utf-8') as f:
        for line in res_line:
            f.write(line + '\n')