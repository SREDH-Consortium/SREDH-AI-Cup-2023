import pandas as pd
import colorama
import re
import os 

file_listdir = os.listdir(r'./data/file/')
all_file_names = []
for file_name in file_listdir:
    if file_name.endswith(".txt"):  # Filter files with a .txt extension
        # delete extension name
        file_name = os.path.splitext(file_name)[0]
        all_file_names.append(file_name)
#print (all_file_names)
all_file_names = all_file_names[800:910]

names=['file_name','label','start','end','content','time']
df = pd.read_csv(rf'./data/answer.txt' , sep='\t', header=None , names=names)

#df.loc[df['time'].notna() , 'content'] = df['content'] + '=>' + df['time'] 
#df = df.drop(columns=['time'])

df['start'] = df['start'].astype(int)
df['end'] = df['end'].astype(int)

#print(df)

#CHECK_FILE_NAME = '11'

for CHECK_FILE_NAME in all_file_names:
    file_path = f'./data/file/{CHECK_FILE_NAME}.txt'
    file_df = df.loc[df['file_name'] == CHECK_FILE_NAME]
    date_df = file_df.loc[file_df['label'] == 'DATE']
    #print(date_df)


    lb = []
    for i in range(len(date_df)):
        word = date_df.iloc[i]['content']
        corr_word = date_df.iloc[i]['time']
        
        
        match1 = re.search(r'(\d{1,2})[/\.](\d{1,2})[/\.](\d{2,4})' , word)
        match2 = re.search(r'(\d{1,2})-(\d{1,2})-(\d{2,4})' , corr_word)

        
        tag = 'dont care'
        if match1 != None and match2 != None:
            
            y1,m1,d1 = int(match1.group(3)) , int(match1.group(2)) , int(match1.group(1))
            y2,m2,d2 = int(match2.group(1)) , int(match2.group(2)) , int(match2.group(3))
            
            if y1 < 100:
                y1 += 2000
            if y2 < 100:
                y2 += 2000
            
            # print(y1,m1,d1)
            # print(y2,m2,d2)
            
            

            if y1 == y2 and m1 == m2 and d1 == d2:
                tag = 'DMY-correct'
            elif y1 == y2 and m1 == d2 and d1 == m2:
                tag = 'MDY-wrong'
            else:
                tag = 'dont care'
        
        lb.append([date_df.iloc[i]['start'] , date_df.iloc[i]['end'] , date_df.iloc[i]['content'] ,corr_word, tag])
        
    #print(lb)    


    print(f'========================    content  {file_path}  ====================')
    content = ''
    with open(file_path , 'r' , encoding='utf-8') as f:
        content = f.read()



    def get_tag(lb , index):
        for i in range(len(lb)):
            if index in range(lb[i][0] , lb[i][1]):
                if index == lb[i][1]-1:
                    return lb[i][4] , '\n'
                return lb[i][4] , ''
        return '' , ''


    for i , ch in enumerate(content):
        tag , end= get_tag(lb , i)
        if tag == 'DMY-correct':
            print(colorama.Fore.GREEN , end='')
        elif tag == 'MDY-wrong':
            print(colorama.Fore.RED , end='')
        elif tag == 'dont care':
            print(colorama.Fore.YELLOW , end='')
        
        else:
            print(colorama.Fore.WHITE , end='')
            #continue
        print(ch , end='')           
        print(end, end='')           
                
                
    input()
    # clear screen
    print('\033c', end='')