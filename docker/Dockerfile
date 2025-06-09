FROM registry.mthreads.com/mcconline/vllm-musa-qy2-py310:v0.7.3
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PATH="/opt/conda/envs/py310/bin:$PATH"
COPY requirements.txt /app/
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 复制后端代码
COPY server /app
