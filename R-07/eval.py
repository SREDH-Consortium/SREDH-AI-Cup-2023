import json
from transformers import HfArgumentParser, TrainingArguments
from src.robust_deid.ner_datasets import DatasetSplitter, DatasetCreator, SpanFixer, SpanValidation
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
# input_file = '/home/ting/code/ehr_deidentification/raw_data/all_valid_data.jsonl'
# ner_dataset_file = '/home/ting/code/ehr_deidentification/First_Phase_Text_Dataset/formal_chunk_size_32_F1_?/ner_datasets/validation.jsonl'
# predictions_file = '/home/ting/code/ehr_deidentification/p_eval_answer/formal_roberta-large_F1_?/predictions.jsonl'
# Initialize the file that will contain the original note text and the de-identified note text
# deid_file = '/home/ting/code/ehr_deidentification/p_eval_answer/formal_roberta-large_F1_?/deid.jsonl'
# If we pass onlpost_processy one argument to the script and it's the path to a json file,
# let's parse it to get our arguments.
model_config = 'eval_AI_cup.json'
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
sequence_tagger.set_eval(
    validation_file=data_args.validation_file,
    max_val_samples=data_args.max_eval_samples,
    preprocessing_num_workers=data_args.preprocessing_num_workers,
    overwrite_cache=data_args.overwrite_cache
)
sequence_tagger.set_eval_metrics(
    validation_spans_file=evaluation_args.validation_spans_file,
    model_eval_script=evaluation_args.model_eval_script,
    ner_types_maps=evaluation_args.ner_type_maps,
    evaluation_mode=evaluation_args.evaluation_mode
)
# sequence_tagger.set_predict(
#     test_file=ner_dataset_file,
#     max_test_samples=data_args.max_predict_samples,
#     preprocessing_num_workers=data_args.preprocessing_num_workers,
#     overwrite_cache=data_args.overwrite_cache
# )

# Initialize the huggingface trainer
sequence_tagger.setup_trainer(training_args=training_args)

# evaluate = sequence_tagger.evaluate()

metrics = sequence_tagger.evaluate()
print(json.dumps(metrics, indent=2))


# predictions = sequence_tagger.predict()
# # Write predictions to a file
# with open(predictions_file, 'w') as file:
#     for prediction in evaluate:
#         file.write(json.dumps(prediction) + '\n')

# text_deid = TextDeid(notation='BILOU', span_constraint='super_strict',output_AI_cup_prediction_file=data_args.output_AI_cup_prediction_file)
# deid_notes = text_deid.run_deid(
#     input_file=input_file,
#     predictions_file=predictions_file,
#     deid_strategy='ai_cup',
#     keep_age=False,
#     metadata_key='meta',
#     note_id_key='note_id',
#     tokens_key='tokens',
#     predictions_key='predictions',
#     text_key='text',
# )
# # Write the deidentified output to a file
# with open(deid_file, 'w') as file:
#     for deid_note in deid_notes:
#         file.write(json.dumps(deid_note) + '\n')