#!/usr/bin/env sh
# 本地启动：浏览器打开 http://localhost:8080
cd "$(dirname "$0")"
echo "Lvan 无痕去水印已启动：http://localhost:8080 （按 Ctrl+C 退出）"
( sleep 1; (xdg-open http://localhost:8080 || open http://localhost:8080) >/dev/null 2>&1 ) &
python3 -m http.server 8080 2>/dev/null || python -m http.server 8080
