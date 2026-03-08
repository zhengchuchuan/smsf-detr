#!/usr/bin/env bash
set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Codex 项目级配置
export CODEX_HOME="$PROJECT_ROOT/.codex"

# API Key（也可以改为从 .env 读取）
export OPENAI_API_KEY="sk-kMRaTBDFM0HcHelxc7NVq3jDK7FIm6s4"

# 启动 Codex
exec codex "$@"
