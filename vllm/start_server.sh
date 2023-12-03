MODEL_NAME_OR_PATH=$1

python3 server.py \
--host 0.0.0.0 \
--port 8000 \
--model $MODEL_NAME_OR_PATH \
--dtype half \
--gpu-memory-utilization 0.5

# usage: server.py [-h] [--host HOST] [--port PORT] [--model MODEL] [--tokenizer TOKENIZER] [--tokenizer-mode {auto,slow}] [--trust-remote-code] [--download-dir DOWNLOAD_DIR]
#                      [--load-format {auto,pt,safetensors,npcache,dummy}] [--dtype {auto,half,bfloat16,float}] [--worker-use-ray] [--pipeline-parallel-size PIPELINE_PARALLEL_SIZE]
#                      [--tensor-parallel-size TENSOR_PARALLEL_SIZE] [--block-size {8,16,32}] [--seed SEED] [--swap-space SWAP_SPACE] [--gpu-memory-utilization GPU_MEMORY_UTILIZATION]
#                      [--max-num-batched-tokens MAX_NUM_BATCHED_TOKENS] [--max-num-seqs MAX_NUM_SEQS] [--disable-log-stats] [--engine-use-ray] [--disable-log-requests]

# optional arguments:
#   -h, --help            show this help message and exit
#   --host HOST
#   --port PORT
#   --model MODEL         name or path of the huggingface model to use
#   --tokenizer TOKENIZER
#                         name or path of the huggingface tokenizer to use
#   --tokenizer-mode {auto,slow}
#                         tokenizer mode. "auto" will use the fast tokenizer if available, and "slow" will always use the slow tokenizer.
#   --trust-remote-code   trust remote code from huggingface
#   --download-dir DOWNLOAD_DIR
#                         directory to download and load the weights, default to the default cache dir of huggingface
#   --load-format {auto,pt,safetensors,npcache,dummy}
#                         The format of the model weights to load. "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors
#                         format is not available. "pt" will load the weights in the pytorch bin format. "safetensors" will load the weights in the safetensors format. "npcache" will
#                         load the weights in pytorch format and store a numpy cache to speed up the loading. "dummy" will initialize the weights with random values, which is mainly
#                         for profiling.
#   --dtype {auto,half,bfloat16,float}
#                         data type for model weights and activations. The "auto" option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
#   --worker-use-ray      use Ray for distributed serving, will be automatically set when using more than 1 GPU
#   --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
#                         number of pipeline stages
#   --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
#                         number of tensor parallel replicas
#   --block-size {8,16,32}
#                         token block size
#   --seed SEED           random seed
#   --swap-space SWAP_SPACE
#                         CPU swap space size (GiB) per GPU
#   --gpu-memory-utilization GPU_MEMORY_UTILIZATION
#                         the percentage of GPU memory to be used forthe model executor
#   --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS
#                         maximum number of batched tokens per iteration
#   --max-num-seqs MAX_NUM_SEQS
#                         maximum number of sequences per iteration
#   --disable-log-stats   disable logging statistics
#   --engine-use-ray      use Ray to start the LLM engine in a separate process as the server process.
#   --disable-log-requests
#                         disable logging requests