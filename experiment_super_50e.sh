# need to be able to import main eraserbenchmark repo
export PYTHONPATH=/home/username/Documents:${PYTHONPATH}

# experiment name (what the model is going to be named)
export EXP_NAME="movie_reproduce_exp_50_supervised"

# CUDA device to use (-1 to use CPU, change to use a different GPU)
export CUDA_DEVICE=0

# output file to save to
export OUTPUT_FILENAME="output_supervised_50e.txt"

dataset_folder=data/movies \
	dataset_name=movies \
	classifier=bert_encoder_generator \
	output_dir=outputs \
	exp_name=$EXP_NAME \
	batch_size=4 \
	rs_weight=1 \
	bash Rationale_model/commands/model_train_script.sh >> ${OUTPUT_FILENAME} 2>&1

dataset_folder=data/movies \
	dataset_name=movies \
	classifier=bert_encoder_generator \
	output_dir=outputs \
	exp_name=$EXP_NAME \
	batch_size=4 \
	bash Rationale_model/commands/model_predict.sh >> ${OUTPUT_FILENAME} 2>&1

python rationale_benchmark/metrics.py \
	--data_dir data/movies \
	--split test \
	--results outputs/bert_encoder_generator/movies/$EXP_NAME/test_prediction.jsonl \
        --score_file outputs/bert_encoder_generator/movies/$EXP_NAME/test_scores.json >> ${OUTPUT_FILENAME} 2>&1
