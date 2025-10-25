@echo off
chcp 65001 > nul
REM ==============================================
REM YOLOv11 预测脚本（支持自定义参数）
REM ==============================================

echo [信息] 启动 YOLOv11 预测任务...

REM 生成时间戳
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do set datestr=%%a%%b%%c
for /f "tokens=1-2 delims=: " %%a in ("%time%") do set timestr=%%a%%b
set timestamp=%datestr%_%timestr%

REM 默认参数
set MODEL_PATH=yolo11x.pt
set SOURCE_PATH=data\test\images

REM 执行预测
python scripts\predict.py ^
    --model "%MODEL_PATH%" ^
    --source "%SOURCE_PATH%" ^
    --name pred_%timestamp%

echo.
echo [信息] 预测结果保存在: runs\predict\pred_%timestamp%
pause
