# need to be able to import main eraserbenchmark repo
export PYTHONPATH=/home/username/Documents:${PYTHONPATH}

# experiment name (what the model is going to be named)
export EXP_NAME="movie_reproduce_exp"

# CUDA device to use (-1 to use CPU, change to use a different GPU)
export CUDA_DEVICE=0
