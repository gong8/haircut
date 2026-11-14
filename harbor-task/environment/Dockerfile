FROM mcr.microsoft.com/playwright:v1.65.0-noble

WORKDIR /opt/haircut
COPY simulator/ ./
COPY haircut-mcp /usr/local/bin/haircut-mcp
RUN npm ci --ignore-scripts && npm install --global opencode-ai@1.17.8 \
    && chmod 755 /usr/local/bin/haircut-mcp && mkdir -p /var/lib/haircut

WORKDIR /workspace
