FROM python:3.12-slim

# The supervisor starts the OpenHands agent sandbox as a sibling container,
# through the host's Docker socket, so it needs the docker CLI (not a daemon).
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app
COPY pyproject.toml README.md ./
COPY supervisor ./supervisor
RUN pip install --no-cache-dir .

ENV SUPERVISOR_CONFIG=/config/supervisor.toml \
    PYTHONUNBUFFERED=1
WORKDIR /projects
ENTRYPOINT ["supervisor"]
CMD ["auto"]
