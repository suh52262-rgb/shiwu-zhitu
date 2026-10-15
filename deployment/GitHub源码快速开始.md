# GitHub源码快速开始

项目首页、功能展示和分级提示示例见[根README](../README.md)。本页用于实际启动，公开源码不带教材、模型、私人媒体和旧题库。

## 只看导览

Windows x64、Python3.11：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-deploy.lock.txt
.\.venv\Scripts\python.exe -m streamlit run frontend/portal_app.py --server.address 127.0.0.1 --server.port 8500
```

打开 http://127.0.0.1:8500 。导览无需密钥和数据库；链接不会自动启动其他页面。

## 完整工作台

按[根README的完整步骤](../README.md#快速开始)，准备.env、自己的API密钥、PostgreSQL、Qdrant与BGE模型；通过scripts/prepare_public_demo.py安装两道AI自编题，或导入自己有权使用的题目。自编题Graph待复核，严格策略不会自动启用未审Graph。

后端启动后，从根目录运行scripts/start_workspace.ps1打开导览及四页。教师默认8502、题库8503、教材8504；学生8501、导览8500、后端8000。

教材页仅管理清单、原文和批次并给导入指引；实际导入使用现有工具。没有教材索引时，不声称提示有教材支持。网络模型可能产生费用。

[首次使用与日常操作](首次使用.md) · [源码发布说明](源码发布说明.md) · [许可与隐私边界](发布许可与隐私边界.md)

完整授权数据包的Restore/Accept只适用于该包，不用于空源码目录。
