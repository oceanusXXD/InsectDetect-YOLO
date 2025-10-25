@echo off
chcp 65001 > nul
REM ==========================================================
REM TensorBoard 可视化服务启动脚本
REM 默认日志目录：runs/train
REM 默认端口：6006
REM 废弃，train代码中修改成自动打开tensorboard
REM ==========================================================

echo [信息] 正在启动 TensorBoard 服务...

REM 设置默认参数
set LOG_DIR=runs/train
set PORT=6006

REM 检查端口占用
netstat -ano | findstr :%PORT% > nul
if %errorlevel% == 0 (
    echo [警告] 端口 %PORT% 已被占用，尝试自动寻找可用端口...
    for /L %%p in (6006,1,6100) do (
        netstat -ano | findstr :%%p > nul
        if %errorlevel% neq 0 (
            set PORT=%%p
            goto port_found
        )
    )
    echo [错误] 6006-6100范围内无可用端口
    pause
    exit /b 1
)
:port_found

REM 启动TensorBoard
echo [信息] 日志目录：%LOG_DIR%
echo [信息] 服务端口：%PORT%
start "" "cmd" /c "tensorboard --logdir %LOG_DIR% --port %PORT% --host 0.0.0.0 --reload_multifile true"

echo.
echo ==================================
echo [信息] TensorBoard 已成功启动！
echo 请访问：http://localhost:%PORT%
echo ==================================
echo 注意：关闭此窗口将终止TensorBoard服务
pause