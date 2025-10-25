@echo off
REM ==========================================================
REM 一键准备 Crop Pests 数据集 (调用 scripts\prepare_data.py)
REM ==========================================================

echo [INFO] 准备运行数据预处理脚本...

REM -------------------------
REM 请根据实际情况修改下面两个变量
REM SOURCE: Kaggle 解压后的原始数据集目录（必填）
REM OUTPUT: 要输出的 YOLO 格式目录（默认 data\crop_pests）
REM -------------------------
set SOURCE=InsectDetect-YOLO\data\raw\crop-pests-dataset
set OUTPUT=InsectDetect-YOLO\data\crop_pests

REM 校验 source 是否存在
if not exist "%SOURCE%" (
    echo [ERROR] 指定的源数据路径不存在: %SOURCE%
    echo 请修改 run_prepare_data.bat 中的 SOURCE 变量，指向你解压后的 Kaggle 数据集目录。
    pause
    exit /b 1
)

echo [INFO] 源路径: %SOURCE%
echo [INFO] 输出路径: %OUTPUT%

REM 执行数据准备脚本
python .\scripts\prepare_data.py --source "%SOURCE%" --output "%OUTPUT%"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 数据准备脚本执行失败，请检查脚本输出与日志。
) else (
    echo [INFO] 数据准备完成，YOLO 数据集保存到: %OUTPUT%
)

pause
