#!/bin/bash

# 1. ライブラリのインストール
echo "--- Installing python libraries ---"
pip install pandas matplotlib huggingface_hub
pip install https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.2/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

apt-get install git-lfs -y
git lfs install

# 2. リポジトリをクローン (カレントディレクトリ内にフォルダが作成されます)
echo "--- Cloning the repository ---"
git clone https://huggingface.co/gitpullpull/Introspective_Temperature_test

echo "--- Setup Complete ---"
ls -ld Introspective_Temperature_test
