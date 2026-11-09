@echo off
chcp 65001 >nul
cd /d %~dp0
"D:\Anaconda3\envs\bearing\python.exe" app\main.py
echo.
echo 程序已退出，按任意键关闭窗口。
pause >nul
