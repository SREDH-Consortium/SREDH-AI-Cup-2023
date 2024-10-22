import json
from src.robust_deid.ner_datasets import DatasetSplitter, DatasetCreator, SpanFixer, SpanValidation
from src.robust_deid.sequence_tagging import SequenceTagger
from src.robust_deid.sequence_tagging.arguments import (
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
)
# Initialize the path where the dataset is located (input_file).
input_train_file = '/home/ting/code/ehr_deidentification/raw_data/all_train_merge_i2b2_data.jsonl'
# Initialize the path where the dataset is located (input_file).
input_vaild_file = '/home/ting/code/ehr_deidentification/raw_data/all_valid_data.jsonl'
# Initialize the location where we will store the train data
train_file_raw = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/train_unfixed.jsonl'
# Initialize the location where we will store the validation data
validation_file_raw = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/validation_unfixed.jsonl'
# Initialize the location where we will store the test data
test_file_raw = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/test_unfixed.jsonl'
# Initialize the location where we will store the train data after fixing the spans
train_file = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/train.jsonl'
# Initialize the location where we will store the validation data after fixing the spans
validation_file = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/validation.jsonl'
# Initialize the location where the spans for hte validation data are stored
validation_spans_file = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/validation_spans.jsonl'
# Initialize the location where we will store the sentencized and tokenized train dataset (train_file)
ner_train_file = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/ner_datasets/train.jsonl'
# Initialize the location where we will store the sentencized and tokenized validation dataset (validation_file)
ner_validation_file = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset_addi2b2/formal_chunk_size_32/ner_datasets/validation.jsonl'
predict_file = '/home/ting/code/ehr_deidentification/raw_data/formal_test_data.jsonl'
ner_predict_file = 'formal_test_data.jsonl'
# Initialize the model config. This config file contains the various parameters of the model.
# model_config = 'train_AI_cup.json'
# Initialize the sentencizer and tokenizer
sentencizer = 'en_core_sci_lg'
tokenizer = 'clinical'
notation = 'BILOU'

spans_key = 'spans'
metadata_key = 'meta'
group_key = 'note_id'
# Create the dataset splitter object
# dataset_train_splitter = DatasetSplitter(
#     train_proportion=100,
#     validation_proportion=0,
#     test_proportion=0
# )
# dataset_valid_splitter = DatasetSplitter(
#     train_proportion=0,
#     validation_proportion=100,
#     test_proportion=0
# )
# dataset_train_splitter.assign_splits(
#     input_file=input_train_file,
#     spans_key=spans_key,
#     metadata_key=metadata_key,
#     group_key=group_key,
#     margin=0.3
# )
# dataset_valid_splitter.assign_splits(
#     input_file=input_vaild_file,
#     spans_key=spans_key,
#     metadata_key=metadata_key,
#     group_key=group_key,
#     margin=0.3
# )
# with open(train_file_raw, 'w') as file:
#     for line in open(input_train_file, 'r'):
#         note = json.loads(line)
#         key = note[metadata_key][group_key]
#         dataset_train_splitter.set_split('train')
#         if dataset_train_splitter.check_note(key):
#             # print(key)
#             file.write(json.dumps(note) + '\n')
# # # Validation split
# with open(validation_file_raw, 'w') as file:
#     for line in open(input_vaild_file, 'r'):
#         note = json.loads(line)
#         key = note[metadata_key][group_key]
#         dataset_valid_splitter.set_split('validation')
#         if dataset_valid_splitter.check_note(key):
#             file.write(json.dumps(note) + '\n')

# with open(validation_spans_file, 'w') as file:
#     for i,span_info in enumerate(SpanValidation.get_spans(
#             input_file=validation_file_raw,
#             metadata_key='meta',
#             note_id_key='note_id',
#             spans_key='spans')):
#         file.write(json.dumps(span_info) + '\n')


# Create the dataset creator object
dataset_creator = DatasetCreator(
    sentencizer=sentencizer,
    tokenizer=tokenizer,
    max_tokens=256,
    max_prev_sentence_token=32,
    max_next_sentence_token=32,
    default_chunk_size=32,
    ignore_label='NA'
)
# ner_types = ["DOCTOR", "DATE", "IDNUM", "MEDICALRECORD", "PATIENT", "HOSPITAL", "TIME", "DEPARTMENT", "CITY", "ZIP", "STREET", "STATE", "ORGANIZATION", "AGE", "DURATION", "SET", "PHONE", "LOCATION-OTHER", "COUNTRY", "URL", "ROOM", "USERNAME", "PROFESSION", "FAX", "DEVICE", "EMAIL", "BIOID", "HEALTHPLAN"]

# # Sometimes there may be some label (span) overlap - the priority list assigns a priority to each label.
# # Higher preference is given to labels with higher priority when resolving label overlap
# ner_priorities = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
# ## Initialize the span fixer object
# span_fixer = SpanFixer(
#     tokenizer=tokenizer,
#     sentencizer=sentencizer,
#     ner_priorities={ner_type: priority for ner_type, priority in zip(ner_types, ner_priorities)},
#     verbose=True
# )
# ## Write the dataset with the fixed validation spans to a file

# with open(train_file, 'w') as file:
#     for note in span_fixer.fix(
#         input_file=train_file_raw,
#         text_key='text',
#         spans_key='spans'
#     ):
#         file.write(json.dumps(note) + '\n')
        
# ner_notes_train = dataset_creator.create(
#     input_file=train_file,
#     mode='train',
#     notation=notation,
#     token_text_key='text',
#     metadata_key='meta',
#     note_id_key='note_id',
#     label_key='label',
#     span_text_key='spans'
# )
# with open(ner_train_file, 'a') as file:
#     for ner_sentence in ner_notes_train:
#         file.write(json.dumps(ner_sentence) + '\n')


# with open(validation_file, 'w') as file:
#     for note in span_fixer.fix(
#         input_file=validation_file_raw,
#         text_key='text',
#         spans_key='spans'
#     ):
#         file.write(json.dumps(note) + '\n')
# # Validation split
# ner_notes_validation = dataset_creator.create(
#     input_file=validation_file,
#     mode='train',
#     notation=notation,
#     token_text_key='text',
#     metadata_key='meta',
#     note_id_key='note_id',
#     label_key='label',
#     span_text_key='spans'
# )
# # Write validation ner split to file
# with open(ner_validation_file, 'w') as file:
#     for ner_sentence in ner_notes_validation:
#         file.write(json.dumps(ner_sentence) + '\n')


ner_notes = dataset_creator.create(
    input_file=predict_file,
    mode='predict',
    notation='BILOU',
    token_text_key='text',
    metadata_key='meta',
    note_id_key='note_id',
    label_key='label',
    span_text_key='spans'
)
# Write to file
with open(ner_predict_file, 'w') as file:
    for ner_sentence in ner_notes:
        file.write(json.dumps(ner_sentence) + '\n')