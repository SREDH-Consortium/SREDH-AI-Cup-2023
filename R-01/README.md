# worker_intelligence_ner
Privacy Protection and Medical Data Standardization Competition: Decoding Clinical Cases, Letting Data Tell the Story
# TEAM_3970
## Competition Results
### Description 
The R-01 team conducted a comparative study to evaluate the efficacy of two approaches: fine-tuning ChatGPT [(based on the GPT-3.5-turbo-1106 model)](https://github.com/TatheerHussain/openai_ner), and a rule-based approach. For the LLM-based method, we pre-processed the training set by segmenting sentences and associating them with the corresponding annotations to generate paragraph-annotation pairs for instruction tuning, following the specifications of the ChatGPT API. The length of each segmented paragraph was limited to 4,096 tokens to meet the API's requirements. After initially fine-tuning the model for three epochs, we adopted a sampling approach to select sentences from the entire training set that contained SHI categories with fewer than 150 instances, such as AGE, ORGANIZATION, COUNTRY, URL, PHONE, DURATION, and SET, as shown in Figure 2. The initial fine-tuned model then underwent a second phase of training on this sampled data, with the goal of improving performance on these underrepresented categories. In contrast, the rule-based approach involved compiling a set of rules and dictionaries collected from online resources and publicly available datasets to recognize and normalize SHIs. We developed rules using regular expressions and employed an analysis tool developed by ourselves to provide a comparative function to assess the disparities between the gold standard training set and the predicted results from the compiled rules. The tool categorized disparities as false positives or false negatives, which were then manually reviewed to refine or modify the patterns until the desired performance was achieved. For SHIs in the ORGANIZATION category, we found that regular expressions were ineffective, so we used a dictionary look-up method with pre-compiled lexicons. For Subtask 2, we specifically developed normalization rules tailored to each date-related SHI subcategory.

### Score
|  User   | Rank  | Subtask 1: Extraction of SHI| Subtask 2: Temporal Information Normalization | 
|  :---:  | :---: | :---------------------:     | :--------------------: |
| zhaorui|3.00 (1)| 0.8660 (5)                  | 0.8685 (1) |

### Detailed Results
| Coding Type | Precision | Recall | F-measure | Support |
|-------------|----------:|:------:|----------:|--------:|
| IDNUM | 0.988775 | 0.9556604 | 0.9719357 | 2120 |
| DOCTOR | 0.9896409 | 0.8614367 | 0.9210992 | 3327 |
| MEDICALRECORD | 0.7879747 | 1 | 0.881416 | 747 |
| PATIENT | 0.9943899 | 0.9902235 | 0.9923022 | 716 |
| DATE | 0.9931213 | 0.9394063 | 0.9655173 | 2459 |
| HOSPITAL | 0.8809723 | 0.8772955 | 0.87913 | 1198 |
| TIME | 0.9954649 | 0.9340426 | 0.9637761 | 470 |
| CITY | 0.9942529 | 0.9276139 | 0.9597781 | 373 |
| STREET | 0.9855072 | 0.9883721 | 0.9869376 | 344 |
| ZIP | 1 | 0.9745042 | 0.9870875 | 353 |
| STATE | 1 | 0.9608434 | 0.9800308 | 332 |
| DEPARTMENT | 0.9280205 | 0.8615752 | 0.8935643 | 419 |
| DURATION | 0.8333333 | 0.8333333 | 0.8333333 | 12 |
| AGE | 0.9591837 | 0.9215686 | 0.9400001 | 51 |
| ORGANIZATION | 0.962963 | 0.7027027 | 0.8125001 | 74 |
| URL | 0 | 0 | 0 | 0 |
| SET | 0.8 | 0.8 | 0.8000001 | 5 |
| LOCATION-OTHER | 1 | 0.3333333 | 0.5 | 6 |
| PHONE | 1 | 1 | 1 | 1 |
| Micro-avg. F| 0.9627724 | 0.9205812 | 0.9412042 | 13007 |
| Macro-avg. F| 0.8996632 | 0.8348374 | 0.8660389 | 13007 |
|-------------|----------:|:------:|----------:|----------:|
|-------------|----------:|:------:|----------:|----------:|
| Temporal Type | Precision | Recall | F-measure | Support |
|-------------|----------:|:------:|----------:|----------:|
| DATE | 0.874026 | 0.8210655 | 0.8467184 | 2459 |
| TIME | 0.8542141 | 0.7978724 | 0.8250825 | 470 |
| DURATION | 1 | 0.8333333 | 0.9090909 | 12 |
| SET | 1 | 0.8 | 0.888889 | 5 |
| Micro-avg.| 0.8715165 | 0.8173795 | 0.8435803 | 2946 |
| Macro-avg.| 0.93206 | 0.8130678 | 0.8685071 | 2946 |
|-------------|----------:|:------:|----------:|----------:|



## Environment

* **Operating System**: Windows 11 Pro Insider Preview
* **CPU**: 12th Gen Intel(R) Core(TM) i9-12900, 2.40 GHz
* **RAM**: 32.0 GB
* **Programming Language**: Python 3.11.5
* **Text Editor**: VSCode 1.84.2


### python library
* pandas 2.0.3


# Instructions for use 

## Competition Output
### clone
```
git clone https://github.com/zhao-rui-NB/worker_intelligence_ner.git
cd worker_intelligence_ner
```


### Generate answer.txt
Manually copy the contents of the `opendid_test` case folder to `worker_intelligence_ner\data\opendid_test`
```
python main.py
```
You can find `answer.txt` in the `.output` folder.


### file_open 
    自動以VSCODE開啟該病例
* -f File Number
* -i Starting Position Index

```
python tool/file_open.py -f 1097 -i 438
```





## ORGANIZATION Dictionary source
### dict V2 : 
* https://www.50pros.com/fortune500
* https://www.sec.gov/files/rules/other/4-460list.htm

