# a program that get command from arg 
# it can open file and set file cousor to that label pos
# -l [753	DURATION	384	391	40/year	P40Y]

import os
import argparse
import subprocess

DATA_DIR = [
    # rf'data\First_Phase\data',
    # rf'data\First_Phase_Validation\data',
    # rf'data\Second_Phase\data',
    rf'data\opendid_test',
    
]

parser = argparse.ArgumentParser()
parser.add_argument('-f' , '--file'  , help='open file')
parser.add_argument('-i' , '--index' , help='set cursor to before index')
parser.add_argument('-l' , '--label' , help='open file and set cursor to label pos')
# keep running
parser.add_argument('-k' , '--keep_run'  , help='keep running' , action='store_true')
args = parser.parse_args()
print(args)


if args.file!=None:
    for dir in DATA_DIR:
        file_path = os.path.join(dir , args.file)
        if '.txt' not in file_path:
            file_path += '.txt'
        if os.path.exists(file_path):
            if args.index==None:
                subprocess.run(['code' , '-n'  , file_path] , shell=1)# open new window 
                break
            else:
                with open(file_path , 'r' , encoding='utf-8') as f:
                    file_content = f.read()
                    line = file_content[:int(args.index)].count('\n')+1
                    char = int(args.index) - file_content[:int(args.index)].rfind('\n')
                
                    subprocess.run(['code' , '-n' , '-g' , f'{file_path}:{line}:{char}'] , shell=1)# open new window # g : goto line:char
                break
                
    else:
        print(f'file {args.file} not found')
        exit()
elif args.label!=None:
    t =  args.label.split()
    print(t)
    file = t[0] 
    start = t[2]
    for dir in DATA_DIR:
        file_path = os.path.join(dir , file)
        if '.txt' not in file_path:
            file_path += '.txt'
        if os.path.exists(file_path):
            # count start pos to line:char
            with open(file_path , 'r' , encoding='utf-8') as f:
                file_content = f.read()
                line = file_content[:int(start)].count('\n')+1
                char = int(start) - file_content[:int(start)].rfind('\n')
            
            subprocess.run(['code' , '-n' , '-g' , f'{file_path}:{line}:{char}'] , shell=1)# open new window # g : goto line:char
            break
    else:
        print(f'file {args.file} not found')
        exit()

elif args.keep_run and args.file==None and args.label==None:
    while True:
        file_name = input('file name : ')
        if file_name=='':
            break
        start = int(input('start pos : '))
        if start=='':
            break


        for dir in DATA_DIR:
            file_path = os.path.join(dir , file_name)
            if '.txt' not in file_path:
                file_path += '.txt'
            if os.path.exists(file_path):
                # count start pos to line:char
                with open(file_path , 'r' , encoding='utf-8') as f:
                    file_content = f.read()
                    line = file_content[:int(start)].count('\n')+1
                    char = int(start) - file_content[:int(start)].rfind('\n')
                
                subprocess.run(['code' , '-n' , '-g' , f'{file_path}:{line}:{char}'] , shell=1)
        # clear least 2 line
        os.system('cls')
        

else:
    print('no file or label')
    print('example: python file_open.py -f 100.txt')
    print('example: python file_open.py -f 100 -i 1000')
    print('example: python file_open.py -l 100.txt 753	DURATION	384	391	40/year	P40Y')
    print()
    
    