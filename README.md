# SREDH/AI CUP 2023


<div align="center">

<!-- Add your project logo if you have any -->
<!-- <img width=200px height=200px src="images/logo.png" alt="Project logo">-->

</div>

<h1 align="center">Patient DeIdentification</h1>

 <p align="center">
 	<!-- Add your tagline or very short intro of your project -->
    Leveraging Large Language Models for De-identification and Temporal Normalization of Sensitive Health Information in Electronic Health Records
<h3 align="center"> Find Detailed Implementation, Summaries and Descriptions Inside Each Project</h3>

<hr />
<br />

<div align="center">

<!-- Add your project demo gif here -->

</div>

<!-- You may write notes in your readme this way if you want to, it looks good and also different from other text -->


<p align="center">Innovative solutions developed by the top 10 teams to tackle the challenge of deidentifying sensitive health information (SHI). This repository showcases cutting-edge approaches leveraging Large Language Models (LLMs), machine learning, and rule-based techniques to ensure robust privacy protection while maintaining the integrity of medical data.</p>

## 🧐 Background

In recent years, artificial intelligence (AI) technology has developed rapidly. Especially in the past year, companies such as OpenAI, Microsoft and Google have introduced and used their own large-scale language models in related products. These applications such as ChatGPT shown the application potential of Large Language Models (LLMs) in various fields. The application of LLM in clinical medicine is therefore regarded as the future of AI in the field of digital health, which is now a very important and evolving research area. However, when applying such AI models, ordinary users and even system or program developers often do not realize the privacy information issues when interacting with LLMs, which may lead to the risk of leaking important confidential information. In addition, if the training data used in training such large language models contains real private information (such as an individual's name, phone number, ID card number, etc.), there is a certain possibility that it will be affected by the memory capacity of the LLM. The ability to interact with users leads to the leakage of private information.


## Competition website link
[Codalab Competition web Link](https://codalab.lisn.upsaclay.fr/competitions/15425?secret_key=db7687a5-8fc7-4323-a94f-2cca2ac04d39)

## 2024-IW-DMRN Workshop link
https://www.sredhconsortium.org/sredh-workshops/2024-iw-dmrn

## 2024-IW-DMRN Workshop Proceedings
https://link.springer.com/book/9789819779659


## 💡 Top-10 Teams Summarries.

The competition attracted a total of 721 participants, constituting 291 teams. Teams played a crucial role in the competition. During the development phase, only the team leader could submit up to three runs per day to evaluate their model performance using the leaderboard. In the final testing phase, each team can submit a maximum of six runs over the course of two days. Out of the participating teams, 103 teams submitted their prediction results for the test set, resulting in 218 submissions during the final testing phase. Since there were 103 teams, the committee invited only the top 10 teams on the leaderboard. 

#### Listed below are descriptions of the top ten teams, labeled R-01 to R-10, with 'R' representing their rank.

## [R-01](https://github.com/TatheerHussain/R-01)
- The R-01 team conducted a comparative study to evaluate the efficacy of two approaches: fine-tuning ChatGPT (based on the GPT-3.5-turbo-1106 model) and a rule-based approach. For the LLM-based method, we pre-processed the training set by segmenting sentences and associating them with the corresponding annotations to generate paragraph-annotation pairs for instruction tuning, following the specifications of the ChatGPT API. The length of each segmented paragraph was limited to 4,096 tokens to meet the API's requirements. After initially fine-tuning the model for three epochs, we adopted a sampling approach to select sentences from the entire training set that contained SHI categories with fewer than 150 instances, such as AGE, ORGANIZATION, COUNTRY, URL, PHONE, DURATION, and SET, as shown in Figure 2. The initial fine-tuned model then underwent a second phase of training on this sampled data, with the goal of improving performance on these underrepresented categories. In contrast, the rule-based approach involved compiling a set of rules and dictionaries collected from online resources and publicly available datasets to recognize and normalize SHIs. We developed rules using regular expressions and employed an analysis tool developed by ourselves to provide a comparative function to assess the disparities between the gold standard training set and the predicted results from the compiled rules. The tool categorized disparities as false positives or false negatives, which were then manually reviewed to refine or modify the patterns until the desired performance was achieved. For SHIs in the ORGANIZATION category, we found that regular expressions were ineffective, so we used a dictionary look-up method with pre-compiled lexicons. For Subtask 2, we specifically developed normalization rules tailored to each date-related SHI subcategory.

## [R-02](https://github.com/TatheerHussain/R-02)
- We utilized the Qwen-14B and 7B models for the subtasks 1 and 2, respectively, based on the findings by Wang, Bi and Zhu, Ren, which indicated that Qwen models perform well in handling clinical notes. To enhance training stability for these large models, we implemented the fully sharded data parallel technique. To mitigate potential overfitting, we employ the NEFTune method, which introduces uniformly distributed noise to the model’s embedding layer. This noise helps prevent the model from memorizing specific details within the training set, allowing it to generalize more effectively to new data. Additionally, this technique reduces the model’s sensitivity to particular inputs, preventing it from developing overly complex representations too early in the training process. Another strategy we employed was randomly extracting sentences within a predefined context window and utilizing them as a basis for validating the model’s performance. The length of the context window was set to 1 for subtask 1, but 3 for subtask 2, as the normalization of temporal information often requires more contextual data. During the inference phase, we used a greedy decoding strategy to ensure the stability and reliability of the model’s output. In addition, we incorporated a rule-based post-processing method since we observed that the fine-tuned model occasionally makes erroneous predictions for some labels that should be straightforward to classify.

## [R-03](https://github.com/TatheerHussain/R-03)
- The R-03 team model architecture is based on the Transformer framework, incorporating GPTNeoXAttention, GPTNeoXRotaryEmbedding, parallel computation of Attention and Feed-Forward, and all Dense Layers. They used the "AdamW" optimization method to enhance the Adam algorithm. Their approach employed a scheduler to adjust the learning rate and selected the "linear warmup" algorithm. Furthermore, they employed rule-based postprocessing to ad-dress low recall rates, particularly for unpopular labels. The team applied rule-based processing to seven labels in the competition and expanded it to include other unpopular and popular la-bels, resulting in improved scores. The study concludes that combining traditional LLM training methods and rule-based postprocessing shows significant promise for future SHI de-identification tasks.

## [R-04](https://github.com/TatheerHussain/R-04)
- The R-04 team developed an algorithm using Inverse Correlational Learning (ICL) and the Sliding Windows method to improve the correlation between sequences and their labels. The sliding windows method understands the start and end of each sequence but could lead to learning irrelevant la-bels. To eliminate these, they introduced a parameter. Furthermore, they used the Fully Sharded Data Parallel (FSDP) technique for stability. Their method proposed two measures to counteract overfitting: the NEFTune method and random extraction of windows. The approach demonstrat-ed outstanding performance in de-identification tasks, with a success rate of over 90%. The COT-few method was integrated into the Prompting approach to address the small number of labels in the DURATION and SET categories.

## [R-05](https://github.com/TatheerHussain/R-05)
- The R-05 team utilized a deep learning model, Pythia-160 m, to train and analyze its performance in a competi-tion. Their training method involved 12 iterations, with AutoModelForCausalLM for consistent text length and a 5e-5 learning rate. They further created a BatchSampler for efficient data split-ting; in their approach, Checkpoint saved weights, and ChatGPT generated synthetic data. Fur-thermore, the data processing included preprocessing and postprocessing. Their results demon-strated that Pythia-160 m performed well in the non-competition dataset, but some scores were lower in the competition set. Despite exploring different models and methods, we chose Pythia-160 m as the prominent architecture.

## [R-06](https://github.com/TatheerHussain/R-06)
- The R-06 team employed a Longformer model in combination with conditional random fields (CRF). Longformer, designed to handle long texts up to 4096 tokens, was chosen to capture the extensive context present in EHRs, a critical feature for accurately identifying SHI. Several pre-processing steps were applied, including tokenization and annotation verification, and we employed a sliding window technique to manage texts exceeding the token limit, ensuring continuity across segments. For model training, we incorporated a fully connected dense layer with 768 units after the Longformer’s transformer output, followed by a CRF layer to improve sequence labelling. The dense layer allowed us to capture rich contextual information, while the CRF layer enhanced the model’s ability to recognize patterns where certain SHI entities tend to appear consecutively. This approach builds on our earlier work in biomedical named entity recognition tasks that emphasize the importance of large contexts to capture the nuances of sensitive information [77]. In the evaluation phase, we used macro-F to assess model performance, ensuring that the model performed well across different PHI categories. Furthermore, to enhance robustness, we implemented a 5-fold ensemble learning approach, where predictions from five distinct models were aggregated using a voting system. This method significantly improved the accuracy and stability of our results, resulting in a final macro F1-score of 0.878 on the testing set. Following the initial model implementation, we integrated a rule-based normalization strategy to address date-related SHIs. Using regular expressions and word-to-number conversion, we normalized textual numbers and date formats. Non-standard entries, such as "several years" or incomplete dates (e.g., "12/18"), were marked as <unknown> to maintain data integrity.

## [R-07](https://github.com/TatheerHussain/R-07)
- The R-07 team used three distinct corpora to identify sensitive personal health information. They used the OpenDeID corpus, the 2014 i2b2/UTHealth de-identification corpus, and the 2016 CEGS N-GRID de-identification corpus. The deidentification process was segmented and tokenized using the Spacy library, and the BIESO tagging scheme was adopted. The Discharge Summary Bi-oBERT hybrid de-identification model demonstrated the ability to maintain a delicate balance between preserving patient privacy and ensuring accurate deidentification. Future research could explore the model's scalability, robustness across diverse settings, and potential integration into health informatics frameworks.

## [R-08](https://github.com/TatheerHussain/R-08)
- The R-08 team proposed method involves a comprehensive approach to preprocessing, model implementation, and rule-based data extraction to achieve effective de-identification of pathology reports. Preprocessing steps included segmenting the text, generating prompts to extract HIPAA-related information, and normalizing temporal expressions. The processed data was formatted to meet the requirements of the Hugging Face Dataset library, enabling seamless integration with advanced learning algorithms. The outcomes were serialized in JSONL format, and postprocessing steps were employed to refine the extracted information, ensuring compatibility for downstream tasks. For text generation, we utilized the T5-Efficient-BASE-DL2 model, an optimized version of the T5 architecture, which offers heightened computational efficiency and reduced memory usage. The model was trained on the preprocessed data using serialized JSONL input-output pairs, with hyperparameters such as a learning rate of 2e-5, a batch size of 4, and 10 training epochs. The evaluation was conducted using the Rouge and Exact Match metrics to assess the model's performance in generating accurate de-identified text. During training, we divided the dataset into training, validation, and test sets. Subsequently, we merged the training and validation sets for final tuning, utilizing a 90-10 split for training and validation, and employed Hugging Faceâ€™s Seq2SeqTrainer to efficiently manage the training process. In addition to the model-based approach, rule-based methods were developed to extract identifiable information such as phone numbers, locations, and city names. These rules included customized regular expressions for local formatting and geographic identifiers, along with a temporal normalization strategy that followed ISO 8601 standards. Temporal expressions, such as "two weeks" or "three months," were systematically converted into structured representations (e.g., "P2W" for two weeks) to ensure consistency and facilitate further analysis. This combination of model-based text generation and rule-based extraction resulted in a robust framework for de-identification, capable of handling both structured and unstructured data effectively.

## [R-09](https://github.com/TatheerHussain/R-09)
- The R-09 team developed a hybrid approach to recognize and standardize Sensitive Health Identifiers (SHIs). The approach used the Pythia language model and a regular expression program to enhance recognition accuracy. The team created a SHI retrieving program that uses regular expression patterns to identify SHIs in medical records. The hybrid approach provides a dataset capture and standardization solution within complex medical records.

## [R-10](https://github.com/TatheerHussain/R-10)
- The R-10 team implemented a longformer model for tasks such as PPIR, NER, and TIN, addressing class distri-bution imbalances and handling overlapping text segments. They further developed a rule-based approach for TIN tasks, addressing limited data and unstructured time-related categories. The study used an ensemble learning technique to integrate five models trained on different datasets, enhancing the robustness and accuracy of entity recognition. Both models showed high preci-sion, recall, and F1 scores across most categories. The Longformer model showed variable per-formance across different categories, suggesting future iterations should focus on category-specific loss weighting and rule-based secondary processing steps.

# Organizers 

- [SREDH Consortium](https://www.sredhconsortium.org/) 

- [National Kaohsiung University of Science and Technology](https://ee.nkust.edu.tw/) 

- [Asia University](https://web.asia.edu.tw/) 


