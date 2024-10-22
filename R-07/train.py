import json
from transformers import HfArgumentParser, TrainingArguments
from src.robust_deid.ner_datasets import DatasetSplitter, DatasetCreator, SpanFixer, SpanValidation
from src.robust_deid.sequence_tagging import SequenceTagger
from src.robust_deid.sequence_tagging.arguments import (
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
)

# Initialize the model config. This config file contains the various parameters of the model.
model_config = 'train_AI_cup.json'
# model_config = 'train_AI_cup_deberta.json'

# import torch
# torch.cuda.set_device(1)
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

# Set the required data for training of the sequence tagger
sequence_tagger.set_train(
    train_file=data_args.train_file,
    max_train_samples=data_args.max_train_samples,
    preprocessing_num_workers=data_args.preprocessing_num_workers,
    overwrite_cache=data_args.overwrite_cache
)

# Set the required data for the evaluation of the sequence tagger
# sequence_tagger.set_eval(
#     validation_file=data_args.validation_file,
#     max_val_samples=data_args.max_eval_samples,
#     preprocessing_num_workers=data_args.preprocessing_num_workers,
#     overwrite_cache=data_args.overwrite_cache
# )
# sequence_tagger.set_eval_metrics(
#     validation_spans_file=evaluation_args.validation_spans_file,
#     model_eval_script=evaluation_args.model_eval_script,
#     ner_types_maps=evaluation_args.ner_type_maps,
#     evaluation_mode=evaluation_args.evaluation_mode
# )

# Initialize the huggingface trainer
sequence_tagger.setup_trainer(training_args=training_args)


# Train the model
# The model is also evaluated every 1000 steps (can be specified in the config file)
sequence_tagger.train(checkpoint=training_args.resume_from_checkpoint)

# metrics = sequence_tagger.evaluate()

# print(json.dumps(metrics, indent=2))
