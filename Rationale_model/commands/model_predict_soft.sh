export data_base_path=${dataset_folder:?"set dataset folder"}

export archive=$output_dir/${classifier:?"set classifier"}/${dataset_name:?"enter dataset name"}/${exp_name:?"set exp name"}
export TEST_DATA_PATH=$data_base_path/test.jsonl

mkdir -p $archive/$saliency

allennlp predict \
--output-file $archive/$saliency/test_prediction.jsonl \
--predictor rationale_predictor \
--include-package Rationale_model \
--silent \
--cuda-device $CUDA_DEVICE \
--batch-size $batch_size \
-o "{model: {saliency_scorer: {type: \"$saliency\", threshold: $threshold}}}" \
--use-dataset-reader \
--dataset-reader-choice validation \
$archive/model.tar.gz $TEST_DATA_PATH