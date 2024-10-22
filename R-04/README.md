# AICUP 隱私保護與醫學數據標準化競賽：解碼臨床病例、讓數據說故事
競賽成績:  
* Rank 4  
* Task 1: 0.8155 / Rank7  
* Task 2: 0.8065 / Rank5
# Introduce
本競賽的目的是將真實醫療院所內已去識別化的電子病歷資料供參賽者進行下列兩項子競賽任務：  
**1. 病患隱私資訊擷取**  
**2. 時間資訊正規化**
# Source code explain
## GPT_data.ipynb
此程式包含兩個功能  
1. 看各個資料的label數量
2. 將chatGPT生成之資料合併原本的dataframe
* chatGPT生成資料的格式說明:  
請chatGPT生成內容以及答案，每筆資料間一定要換行，內容和答案間要有一個'\t'符號，若有複數個答案，要在答案間加上一個'\n'  
例: _P.O. BOX 76\tLOCATION-OTHER: P.O. BOX 76_

## Final_training.ipynb
訓練模型的程式碼  
* training dataset path: 將Training Dataset中的 `data_path = your_training_dataset_path` 改為自己訓練集的路徑即可  
* model save path: 將Training的 `torch.save(your_model_save_path)` 換成你想存放的雲端位置

## add_dataset.ipynb
將原始資料夾的文字檔轉成dataframe，並存成tsv檔&nbsp;`txt_to_trainingData(txt_path, ans_path)`  
txt_path為文字檔所在位置，ans_path為answer檔所在位置

## test_data.ipynb
inference測試資料輸出結果
* training dataset path: 將Load test dataset中的 `folder_path = your_test_dataset_path` 改為自己測試集的路徑即可
* model path: 將Model載入中的 `save_path = your_load_model_path` 改為自己模型的路徑即可
* answer.txt: 本程式最終輸出的答案檔為 _process4_ans.txt_，請再把它重新命名為 _answer.txt_

