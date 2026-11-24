FROM python:3.12-slim

# The supervisor starts the OpenHands agent sandbox as a sibling container,
# through the host's Docker socket, so it needs the docker CLI (not a daemon).
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

WORKDIR /app

# 1. Dependencies (the OpenHands SDK and friends: the slow part). Installed
#    from pyproject.toml alone, with an empty stand-in package, so this layer
#    is reused on every rebuild until pyproject.toml changes. The pip cache
#    lives outside the image (BuildKit cache mount), so even then only new or
#    changed packages are downloaded, and the image stays small.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir supervisor && touch supervisor/__init__.py \
    && pip install . \
    && rm -rf supervisor build

# 2. The supervisor's own code: a few seconds, rebuilt on every code change.
COPY supervisor ./supervisor
RUN pip install --no-cache-dir --no-deps --force-reinstall .

ENV SUPERVISOR_CONFIG=/config/supervisor.toml \
    GOAL_FILE=/config/goal.md \
    PYTHONUNBUFFERED=1
WORKDIR /projects
EXPOSE 8765
ENTRYPOINT ["supervisor"]
CMD ["web"]
