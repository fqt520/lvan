@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Lvan 无痕去水印已启动：http://localhost:8080 （关闭此窗口即退出）
start "" http://localhost:8080
python -m http.server 8080 || py -m http.server 8080
pause
