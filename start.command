#!/bin/zsh
cd -- "$(dirname "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print "请先按 README 创建 .venv 并安装依赖。"
  exit 1
fi
exec .venv/bin/python -m musictracker start
