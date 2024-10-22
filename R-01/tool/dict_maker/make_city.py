import pandas as pd


ANSWER_PATH = rf'./data/answer.txt'
OUTPUT_PATH = rf'./dictionary/CITY_dict.txt'



names=['file_name','label','start','end','content','time']
df = pd.read_csv(ANSWER_PATH , sep='\t', header=None , names=names)
city_df = df[df['label']=='CITY']
#cnt city_df type
city_cnt = city_df['content'].value_counts()
# get all different city
all_city = city_cnt.index.tolist()
print(all_city)


#save all_city to file
with open(OUTPUT_PATH, 'w' , encoding='utf-8') as f:
    for city in all_city:
        f.write(city+'\n')




