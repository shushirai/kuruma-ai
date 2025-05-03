# ✅ ベースイメージ（CUDA + cuDNN + Python環境付き）
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# ✅ 非対話モード
ENV DEBIAN_FRONTEND=noninteractive

# ✅ 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git curl sudo ffmpeg libgl1 unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ✅ シンボリックリンクで `python` / `pip` を使えるように
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# ✅ 一般ユーザーを作成（ホストとUID合わせるならビルド引数化可能）
ARG USERNAME=shu_Docker
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ✅ 作業ディレクトリを指定（ホストと共有される）
WORKDIR /workspace

# ✅ ユーザーを切り替え
USER ${USERNAME}

# ✅ Python 仮想環境構築（任意。venvで切り分けたい場合）
# RUN python -m venv venv && . venv/bin/activate

# ✅ requirements.txt を使った依存ライブラリのインストール（後でCOPYしてもOK）
# COPY requirements.txt .
# RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["/bin/bash"]
