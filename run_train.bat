@echo off
chcp 65001 > nul
REM ==========================================================
REM 运行训练（只运行脚本，不传参数）
REM ==========================================================


REM 检查脚本是否存在
if not exist ".\scripts\train.py" (
    echo [ERROR] 找不到 scripts\train.py！
    pause
    exit /b 1
)

echo [INFO] run scripts\train.py...
python .\scripts\train.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] 训练脚本执行发生错误，退出码: %ERRORLEVEL%
) else (
    echo [INFO] 训练任务已完成！
)

pause
