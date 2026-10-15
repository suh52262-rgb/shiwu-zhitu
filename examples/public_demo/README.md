# 自编公开演示材料 v1

两道题、参考解和 Graph 均由 AI 于 2026-10-15 自编，不复制本地教材、成员题库、截图或真实学生输入。仅作工程演示，不计入历史 T4、留出集或教学效果证据。这些项目自编样例随源码采用AGPL-3.0-only，见根目录LICENSE；不宣称获得任何教材分发权。

|题号|内容|参考结果|
|---|---|---|
|public_force_01|相反方向的水平力、合力、加速度和末速度|8 N向右、4 m/s²向右、12 m/s向右|
|public_impulse_02|速度反向、冲量和平均合力；允许两条等价路线|2 N·s向左、10 N向左|

在项目根目录运行：

```powershell
.venv/Scripts/python.exe scripts/prepare_public_demo.py
.venv/Scripts/python.exe examples/public_demo/make_sample_media.py --video
```

第一条只新增上述题号；相同内容重复运行不改写，任何冲突会在写入前拒绝。第二条在 `output/public_demo_media` 生成四张排版图片和一段24秒视频：前三张逐步增加书写内容，第四张故意将方向相反的力相加。**这是排版的合成输入，不是学生手写或真实解题录像；8秒画面保持不是认知卡顿。** 再次生成时请指定新的 `--output` 目录。

启动后在学生端搜索题号，先用 `public_force_01` 上传 `force_stage_1.png`，再请求提示或以文字续答。另开一轮上传 `force_error.png` 检查错误提示。视频演示上传 `force_process.mp4`，核对三个书写阶段与时间来源。网络教学仍需用户自己的有效API密钥，会产生费用；生成材料本身不调用模型。

所有 Graph 均标记 AI 编写、人工未复核；只有 `GRAPH_RELEASE_POLICY=legacy-demo` 才作为未复核演示图使用。`published-only` 不会把样例当作教师已批准。规则校验与数值检查不代表完整教学语义验收。

不附教材或向量库。缺少教材时，教材清单为空、RAG可能失败或无命中，不能说已有教材支持。后续可自行导入有权使用的教材，本样例不保证RAG充分性。样例也不改变已有私有题库。
