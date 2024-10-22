# AI CUP 2023-TEAM-3868
隨著醫學科技的快速發展，醫療領域的數據應用變得越來越重要。這些數據不僅有助於臨床醫生更好地理解病患的健康狀況，還可以用於疾病預測、醫療研究以及醫療政策的制定。然而，這種數據的使用也帶來了一個關鍵問題，即隱私保護。如何在充分利用醫學數據的同時確保病患的隱私成為了一個亟待解決的挑戰。

此 project 為 AI CUP 隱私保護與醫學數據標準化競賽：解碼臨床病例、讓數據說故事比賽的 Github code，程式部分我們有參考 [Robust DeID: De-Identification of Medical Notes using Transformer Architectures](https://github.com/obi-ml-public/ehr_deidentification/tree/main) 的訓練方式，並且設計了獨特的 PostProcess System，資料的部分我們還增加了 [i2b2 資料集](https://portal.dbmi.hms.harvard.edu/)，讓 label 從原本的 21 個增加至 28 個，增添了一定的泛化能力，更多細部請看[DOCS](https://docs.google.com/document/d/1dTvvhDPMTLfh49-yTl7hSe3SgdtqZJ2VHw4hNgfVkjQ/edit?usp=sharing)。
## 運行環境
### 使用 Anaconda 建立環境
```bash
conda create -n ehr python=3.7
```
### Python 套件
```bash
pip install -r requirements.txt
```
## Checkpoint
|        Class         | Description                          
| :------------------: | :----------------------------------- 
|     `formal_split`          | [huggingface](https://huggingface.co/vickt/AI_CUP_deidentification_formal_split/tree/main)
|     `Add_i2b2_formal_split`          | [huggingface](https://huggingface.co/vickt/AI_CUP_deidentification_add_i2b2_formal_split/tree/main)
|     `Add_i2b2_lower_continue`          | [huggingface](https://huggingface.co/vickt/AI_CUP_deidentification_add_i2b2_lower_continue/tree/main)
## 運行方式
請新增 ```raw_data/``` folder 在路徑下並執行 ```data_create.ipynb``` 中的程式，程式將會創建 Train Test Dev dataset ，其格式為:
### Train Dev data format
```python3
{
    "text": '...', // document content
    'meta': {"note_id": file_name},
    'spans': [{'label': 'MEDICALRECORD' ...}],
}
```
### Test data format
```python3
{
    "text": '...', // document content
    'meta': {"note_id": file_name},
    'spans': [],
}
```
