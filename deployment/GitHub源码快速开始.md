# 视物知途 · GitHub源码候选（本地演示）

大学物理过程辅导工作台：照片分级提示、视频书写过程复盘，以及教师证据、题库与教材管理。AI报告是需核对的分析，不能把画面停留当作学生认知事实。

Windows用户可先双击根目录 **首次使用.cmd**，查看说明、检查环境、安装依赖或只打开导览。菜单不会自动恢复教材或调用付费模型，详细操作见[首次使用](deployment/首次使用.md)。源码包缺少数据是有意排除，并非安装失败。

[GitHub占位仓库](https://github.com/suh52262-rgb/shiwu-zhitu)已创建，当前仅有简介；本源码候选尚未上传。维护者已选择原创源码采用AGPL-3.0-only，保留当前PDF实现；见根目录LICENSE和NOTICE。第三方依赖、模型和教材不能用一个项目许可证统一替代，见[第三方声明](THIRD_PARTY_NOTICES.md)及[许可边界](deployment/发布许可与隐私边界.md)。

**测试边界：离线回归已迁移到tests/fixtures/independent的合成夹具，不读取私有题库或教材金标。** 兼容题号仅是旧软件接口；合成的300条检索记录与72/36/108计数用例不是RAG质量或T4新评测。运行 `TEST_OFFLINE=1` 并按CI选集检查，严格门槛不允许普通skip。最终受检提交与结果须以对应候选验收记录为准，不能沿用旧包或工作区的绿灯。仍需区分软件测试、异机安装与真实教学质量验收。

支持范围：Windows x64、Python3.11。Linux CI配置存在，但本次没有远端运行或跨平台部署结论。公开源码候选包不带教材、原demo题库/配图、密钥、模型或学生历史，不是完整数据部署包。仅附 `examples/public_demo` 中两道AI自编题及生成输入的工具，来源和许可边界见该目录README。

## 最小页面体验

```powershell
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-deploy.lock.txt
.venv/Scripts/python.exe -m streamlit run frontend/portal_app.py --server.address 127.0.0.1 --server.port 8500
```

仅导览无需数据库和密钥；它提供四页入口，但入口链接不会自动启动其他服务。完整工作台需继续准备后台。

## 教学工作台

先安装附带的自编题和可选合成媒体：

```powershell
.venv/Scripts/python.exe scripts/prepare_public_demo.py
.venv/Scripts/python.exe examples/public_demo/make_sample_media.py --video
```

这不会改写原题库，冲突时会拒绝；生成图片和视频仅为AI排版演示输入，不是真实学生材料。Graph保留未人工复核标记；严格发布策略下不会自动启用。没有教材时不宣称已获得RAG教材支持。

复制.env.example为.env，按实际环境配置PostgreSQL、Qdrant、本地BGE路径及自己的API密钥。不要提交.env。根目录compose配置对应本地数据服务；首次Docker镜像下载需网络。执行scripts/init_db.py初始化新库，再启动：

```powershell
.venv/Scripts/python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

另一个终端运行scripts/start_workspace.ps1，启动导览和四页。当前题库为空时不是损坏：先从题库导入页导入自己有权使用的题目、核对题面并发布，再独立复核Graph。教材实际导入依赖现有工具与BGE索引，参见scripts中的CLI帮助；未导入时不宣称RAG已有教材依据。

## 完整授权本地包

获取完整本地包后按快速部署执行Setup → Preflight → Restore → Accept → Start。源码模式无manifest和教材导出，不能直接执行该包的Restore/Accept流程。完整包不是可自动公开的GitHub附件。

公开前仍需审核源码、文档、仓库历史及题库材料许可；本次只生成本地源码候选ZIP，尚未向占位仓库上传源码。网络模型会产生费用，离线页面浏览不代表在线教学可用。
