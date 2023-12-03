MODEL_NAME_OR_PATH=$1
PORT=$2
# docker build -t vllm:0.2.2 -f Dockerfile .

# Docker can run your container in detached mode in the background.
# To do this, you can use the --detach or -d for short

docker run \
--gpus all \
--rm \
--shm-size=20g \
-p $2:8000 \
-v $(pwd):/app \
-v $MODEL_NAME_OR_PATH:/model_repo \
vllm:0.2.2 bash -c "cd /app && ./start_server.sh /model_repo"