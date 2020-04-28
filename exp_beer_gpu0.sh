export PYTHONPATH=/home/username/Documents:${PYTHONPATH}
export EXP_NAME="beer_u_e5_l1.0"
export MODEL_NAME="bert_encoder_generator_beer_5e_l1.0"
export CUDA_DEVICE=0
export OUTPUT_FILENAME="oo_${EXP_NAME}.txt"

#dataset_folder=data/beer \
#	dataset_name=beer \
#	classifier=$MODEL_NAME \
#	output_dir=outputs \
#	exp_name=$EXP_NAME \
#	batch_size=8 \
#	rs_weight=0 \
#	bash Rationale_model/commands/model_train_script.sh >> ${OUTPUT_FILENAME} 2>&1

#cp -r outputs/${MODEL_NAME}/beer outputs/${MODEL_NAME}/beermini

dataset_folder=data/beermini \
	dataset_name=beermini \
	classifier=$MODEL_NAME \
	output_dir=outputs \
	exp_name=$EXP_NAME \
	batch_size=8 \
	bash Rationale_model/commands/model_predict.sh >> ${OUTPUT_FILENAME} 2>&1

python rationale_benchmark/metrics.py \
	--data_dir data/beermini \
	--split test \
	--results outputs/${MODEL_NAME}/beermini/$EXP_NAME/test_prediction.jsonl \
	--score_file outputs/${MODEL_NAME}/beermini/$EXP_NAME/test_scores.json >> ${OUTPUT_FILENAME} 2>&1
