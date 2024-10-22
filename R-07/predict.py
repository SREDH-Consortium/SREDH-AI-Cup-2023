import json
from transformers import HfArgumentParser, TrainingArguments
from src.robust_deid.ner_datasets import DatasetCreator
from src.robust_deid.sequence_tagging import SequenceTagger
from src.robust_deid.sequence_tagging.arguments import (
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
)
from src.robust_deid.deid import TextDeid

parser = HfArgumentParser((
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
    TrainingArguments
))
# Initialize the path where the dataset is located (input_file).
# Input dataset
input_file = 'raw_data/formal_test_data.jsonl'
# Initialize the location where we will store the sentencized and tokenized dataset (ner_dataset_file)
ner_dataset_file = 'TEST_preprocess_data/formal_test_data.jsonl'
# Initialize the location where we will store the model predictions (predictions_file)
# Verify this file location - Ensure it's the same location that you will pass in the json file
# to the sequence tagger model. i.e. output_predictions_file in the json file should have the same
# value as below
predictions_file = 'formal_test/formal_chunk_size_32/predictions.jsonl'
# Initialize the file that will contain the original note text and the de-identified note text
deid_file = 'formal_test/formal_chunk_size_32/deid.jsonl'
# Initialize the model config. This config file contains the various parameters of the model.
model_config = 'predict_AI_cup.json'


# Create the dataset creator object
# dataset_creator = DatasetCreator(
#     sentencizer='en_core_sci_lg',
#     tokenizer='clinical',
#     max_tokens=256,
#     max_prev_sentence_token=32,
#     max_next_sentence_token=32,
#     default_chunk_size=32,
#     ignore_label='NA'
# )

# ner_notes = dataset_creator.create(
#     input_file=input_file,
#     mode='predict',
#     notation='BILOU',
#     token_text_key='text',
#     metadata_key='meta',
#     note_id_key='note_id',
#     label_key='label',
#     span_text_key='spans'
# )
# # Write to file
# with open(ner_dataset_file, 'w') as file:
#     for ner_sentence in ner_notes:
#         file.write(json.dumps(ner_sentence) + '\n')
        
parser = HfArgumentParser((
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
    TrainingArguments
))
# If we pass only one argument to the script and it's the path to a json file,
# let's parse it to get our arguments.
model_args, data_args, evaluation_args, training_args = parser.parse_json_file(json_file=model_config)
# Initialize the sequence tagger
sequence_tagger = SequenceTagger(
    task_name=data_args.task_name,
    notation=data_args.notation,
    ner_types=data_args.ner_types,
    model_name_or_path=model_args.model_name_or_path,
    config_name=model_args.config_name,
    tokenizer_name=model_args.tokenizer_name,
    post_process=model_args.post_process,
    cache_dir=model_args.cache_dir,
    model_revision=model_args.model_revision,
    use_auth_token=model_args.use_auth_token,
    threshold=model_args.threshold,
    do_lower_case=data_args.do_lower_case,
    fp16=training_args.fp16,
    seed=training_args.seed,
    local_rank=training_args.local_rank
)
# Load the required functions of the sequence tagger
sequence_tagger.load()

# Set the required data for the evaluation of the sequence tagger
sequence_tagger.set_predict(
    test_file=ner_dataset_file,
    max_test_samples=data_args.max_predict_samples,
    preprocessing_num_workers=data_args.preprocessing_num_workers,
    overwrite_cache=data_args.overwrite_cache
)
sequence_tagger.setup_trainer(training_args=training_args)

predictions = sequence_tagger.predict()
# Write predictions to a file
with open(predictions_file, 'w') as file:
    for prediction in predictions:
        file.write(json.dumps(prediction) + '\n')

text_deid = TextDeid(notation='BILOU', span_constraint='super_strict',output_AI_cup_prediction_file=data_args.output_AI_cup_prediction_file)
deid_notes = text_deid.run_deid(
    input_file=input_file,
    predictions_file=predictions_file,
    deid_strategy = data_args.deid_strategy,
    keep_age=False,
    metadata_key='meta',
    note_id_key='note_id',
    tokens_key='tokens',
    predictions_key='predictions',
    text_key='text',
)
# Write the deidentified output to a file
with open(deid_file, 'w') as file:
    for deid_note in deid_notes:
        file.write(json.dumps(deid_note) + '\n')