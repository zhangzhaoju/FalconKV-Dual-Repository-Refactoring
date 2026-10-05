ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG PYTHON=/usr/local/python3.11.14/bin/python
ENV PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
COPY wheels/ /opt/paired-release/wheels/
COPY image-input.json install_image.py check_pip_dependencies.py /opt/paired-release/
RUN ${PYTHON} -B /opt/paired-release/install_image.py
WORKDIR /opt/paired-release
CMD ["/bin/bash"]
