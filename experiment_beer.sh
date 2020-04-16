export PYTHONPATH=/home/username/Documents:${PYTHONPATH}
export EXP_NAME="beermini_50_u"
export CUDA_DEVICE=0
export OUTPUT_FILENAME="output_beermini_50_u.txt"

dataset_folder=data/beermini \
	dataset_name=beermini \
	classifier=bert_encoder_generator_beer \
	output_dir=outputs \
	exp_name=$EXP_NAME \
	batch_size=1 \
	rs_weight=0 \
	bash Rationale_model/commands/model_train_script.sh >> ${OUTPUT_FILENAME} 2>&1
