# Frozen environment adaptation of OfficeBench docker/Dockerfile at
# b978b808667c32b52ce19a67ce1def1de9ae02b7 (Apache-2.0).
# No credentials or benchmark answers are included. Fresh containers replace
# upstream root-filesystem git reset. Explicitly reported as an execution port.
FROM docker.m.daocloud.io/library/ubuntu@sha256:076e7bf82319619f28dd0aa9c02ab34b055ad37100395b4f49a2ffba04fb5e11
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g;s@http://security.ubuntu.com/ubuntu/@http://mirrors.tuna.tsinghua.edu.cn/ubuntu/@g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates bash coreutils findutils tree git \
    libreoffice tesseract-ocr libtesseract-dev fonts-dejavu fonts-liberation && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /requirements.txt
RUN python3 -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r /requirements.txt && \
    python3 -m pip freeze > /environment-pip-freeze.txt && \
    dpkg-query -W > /environment-dpkg-versions.txt
WORKDIR /
LABEL idea4.officebench.commit="b978b808667c32b52ce19a67ce1def1de9ae02b7"
