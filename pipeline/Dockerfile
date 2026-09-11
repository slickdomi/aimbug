# Data pipeline image: turns the MaleCNS v1.0 flat connectome into compact
# binaries for the browser. Runs as container root (rootless docker maps it
# to the host user).
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

# Dependency cooldown (https://cooldowns.dev/): nothing uploaded after this date.
ARG EXCLUDE_NEWER=2026-08-28

RUN pip install --no-cache-dir uv==0.8.0 \
 && uv pip install --system --no-cache --exclude-newer "${EXCLUDE_NEWER}" \
      numpy pandas pyarrow scipy pillow

WORKDIR /work
