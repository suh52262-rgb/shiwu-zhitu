@echo off
rem 视物知途 · 学生端一键启动（双击这个文件）
rem
rem 逻辑在 scripts\start_student_app.ps1 —— 用 PowerShell 而不是纯 cmd：
rem   cmd 里 `set /a` 在 if 块中不实时生效、`timeout` 在重定向下会失败，
rem   第一版纯 cmd 脚本实测两个坑都踩了。
rem
rem 加参数（如 -NoBrowser）请直接调 ps1：
rem   powershell -ExecutionPolicy Bypass -File scripts\start_student_app.ps1 -NoBrowser

chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start_student_app.ps1" %*
if errorlevel 1 pause
