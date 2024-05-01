conda create -n MGLE python=3.10 -y
conda activate MGLE

pip install -U pip cmake cython==0.29.36 pydantic==1.10 numpy
pip install -U gdown pydrive2 wget jupyter jupyterlab jupyterthemes ipython
pip install -U sentencepiece transformers diffusers tokenizers datasets gradio==3.37 accelerate evaluate git+https://github.com/openai/CLIP.git
pip install -U https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-cp310-cp310-linux_x86_64.whl https://download.pytorch.org/whl/cu113/torchvision-0.13.0%2Bcu113-cp310-cp310-linux_x86_64.whl https://download.pytorch.org/whl/cu113/torchaudio-0.12.0%2Bcu113-cp310-cp310-linux_x86_64.whl
pip install -U deepspeed

pip install torch==2.0.1 torchvision==0.15.2
git submodule update --init --recursive
cd LLaVA
pip install -e .
# pip install -U https://download.pytorch.org/whl/cu113/torch-1.12.0%2Bcu113-cp310-cp310-linux_x86_64.whl https://download.pytorch.org/whl/cu113/torchvision-0.13.0%2Bcu113-cp310-cp310-linux_x86_64.whl https://download.pytorch.org/whl/cu113/torchaudio-0.12.0%2Bcu113-cp310-cp310-linux_x86_64.whl
pip install -U ninja flash-attn==1.0.2
pip install -U pydrive2 gdown wget

# python3 -m llava.model.apply_delta \
#     --base huggyllama/llama-7b \
#     --target _ckpt/LLaVA-7B-v1 \
#     --delta liuhaotian/LLaVA-Lightning-7B-delta-v1-1

cd ..

cp mgie_llava.py LLaVA/llava/model/llava.py
cp mgie_train.py LLaVA/llava/train/train.py
