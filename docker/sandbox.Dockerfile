# Image for DockerSandbox: the agent's commands and the hidden tests run here with --network none,
# so everything a repo's test_cmd needs must be baked in at build time.
#   docker build -f docker/sandbox.Dockerfile -t repo-fix-sandbox:py3.12 .
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "pytest>=8"
# DockerSandbox runs commands as the host uid (so files in the bind mount stay owned by the host
# user); give that arbitrary uid a writable HOME and let git trust the mounted repo.
ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1 GIT_TERMINAL_PROMPT=0
RUN git config --system safe.directory '*'
WORKDIR /workspace/repo
