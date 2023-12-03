# GalacTex Server for Text Generation

This repo contains the code for the server that runs the Galactica model for text generation. It is based on the [VLLM](https://github.com/vllm-project/vllm) project, *and soon I will add support for the [Hugging Face Text Generation Inference](https://github.com/huggingface/text-generation-inference) codebase.*

Compatible with all Galactica models from [here](https://huggingface.co/models?search=galactica) and all models supported by the VLLM project.

Use the [GalacTex Chrome Extension](https://chromewebstore.google.com/detail/galactex/mkkbiefcllablljdpkmcgmjkikobemjb) to use the server with Overleaf.

## Model Download

Download the model from [here](https://huggingface.co/models?search=galactica) using:

```bash
huggingface-cli download --repo-type model --local-dir "<LOCAL_MODEL_REPO>/facebook--galactica-6.7b" --local-dir-use-symlinks False "facebook/galactica-6.7b"
```

## Usage

### Docker (Recommended)
```bash
cd vllm
docker build -t vllm:0.2.2 -f Dockerfile .
./start_server_docker.sh <MODEL_PATH> <PORT>
```
### Local Installation

Follow the instructions on the [VLLM Docs](https://vllm.readthedocs.io/en/latest/getting_started/installation.html) to install the VLLM server.
Run the VLLM server with:

```bash
./run_api_server.sh <MODEL_PATH>
```

Then go to the extension setting and set the server endpoint to `http://localhost:<PORT>/api/generate`.

# Motivation

This is a quick and dirty project to make writing papers with code completion easier. Born out of procrastination and the need to quickly write a paper during my PhD.

# Contacts
**Wissam Antoun**: [Linkedin](https://www.linkedin.com/in/wissam-antoun-622142b4/) | [Twitter](https://twitter.com/wissam_antoun) | [Github](https://github.com/WissamAntoun) |  wissam.antoun (AT) gmail (DOT) com |  wissam.antoun (AT) inria (DOT) fr