# 第三方声明与许可证

本项目原创源码采用 **AGPL-3.0-only**（维护者已于2026-10-15明确选择），完整条款见[LICENSE](LICENSE)，范围与署名见[NOTICE](NOTICE)。第三方组件保留各自许可证，不能把整张依赖表改标为AGPL。

本清单对应[Windows/Python3.11精确依赖锁](deployment/requirements-windows-py311.lock.txt)的115项分发，加pip安装工具，共116项。已安装版本与锁逐项一致；额外安装的PPT等开发工具不冒充运行依赖。完整元数据、许可证原文件及SHA256见[清单](third_party/inventory.json)。保留原文件与上游补充文本不等同穷尽原生库、专利或各项源代码提供义务。

## 关键处理

- **PyMuPDF 1.28.2**：保留其AGPL/Artifex商业双许可说明；本项目使用AGPL路径，没有购买或声称获得商业授权。附完整AGPL文本，保留现有PDF实现。[官方说明](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)。
- **psycopg / psycopg-binary**：保留LGPL及包内相关声明。二进制实现包含客户端库，不能仅看Python接口的许可就认为覆盖其全部组件。[官方安装说明](https://www.psycopg.org/psycopg3/docs/basic/install.html)。
- **NumPy、SciPy、PyTorch、OpenCV**等：保留随分发提供的第三方许可与声明；多许可证字段按原样记录，不概括成单一MIT。
- **LangSmith、Streamlit、Tokenizers**：已从各自官方仓库的对应版本标签补取缺失文本，来源和哈希见[补充来源](third_party/upstream_sources.json)。
- **BGE模型**：公开源码候选不携带权重。既有固定修订模型卡声明MIT；下载或另行分发时应查阅其原始模型卡及适用声明。这里不把模型授权文本补造为本项目授权。
- **DeepSeek等远程服务**：调用代码的许可证不授予模型服务使用权；服务账号和调用遵守供应商条款。源码不带API密钥或模型权重。
- **PostgreSQL、Qdrant、Docker**：源码含部署配置，不内嵌数据库或Docker程序；各自软件/镜像及Docker Desktop适用条款另行保留、核查。
- **教材、外部题库与配图、私人视频、学生记录**：不因源码采用AGPL而获得分发权，继续排除。项目自编示例和合成测试夹具已明确标注；不冒充真实学生或教师研究。

## 重建与核对

用对应锁环境运行 `scripts/collect_third_party_licenses.py --lock deployment/requirements-windows-py311.lock.txt --output <新目录> --supplements third_party/upstream_sources.json`。只读取本机已安装元数据和已保留文本，不安装依赖、不联网；版本不一致或上游补充哈希不符会失败。此处的覆盖指“116项均有保留文本/声明”，不是全依赖法律合规认证。

旧1.4声明保留于内部历史归档，旧部署包不因本轮文本变更自动获得新的发行或材料授权。根目录LICENSE不表示GitHub已经公开。

## 逐项记录

|依赖|版本|原始许可标记（简写）|保留文件数|
|---|---|---|---|
|altair|6.3.0|Copyright (c) 2015-2025, Vega-Altair Developers|1|
|annotated-doc|0.0.5|MIT|1|
|annotated-types|0.8.0|MIT|1|
|anyio|4.15.1|MIT|1|
|attrs|26.1.0|MIT|1|
|certifi|2026.7.22|MPL-2.0|1|
|charset-normalizer|3.5.2|MIT|1|
|click|8.5.0|BSD-3-Clause|1|
|cloudpickle|3.1.2|BSD-3-Clause|1|
|colorama|0.4.6|License :: OSI Approved :: BSD License|1|
|distro|1.9.0|Apache License, Version 2.0|1|
|fastapi|0.142.1|MIT|1|
|filelock|4.0.7|MIT|1|
|fsspec|2026.9.0|BSD-3-Clause|1|
|grpcio|1.84.0|Apache-2.0|1|
|h11|0.16.0|MIT|1|
|h2|4.4.1|MIT|1|
|hf-xet|1.6.0|Apache-2.0|1|
|hpack|4.2.0|MIT|1|
|httpcore|1.0.9|BSD-3-Clause|1|
|httpcore2|2.13.1|BSD-3-Clause|1|
|httptools|0.8.0|MIT|3|
|httpx|0.28.1|BSD-3-Clause|1|
|httpx2|2.13.1|BSD-3-Clause|1|
|huggingface_hub|1.33.0|Apache-2.0|1|
|hyperframe|6.1.0|The MIT License (MIT)|1|
|idna|3.20|BSD-3-Clause|1|
|iniconfig|2.3.0|MIT|1|
|itsdangerous|2.2.0|License :: OSI Approved :: BSD License|1|
|Jinja2|3.1.6|License :: OSI Approved :: BSD License|1|
|jiter|0.17.0|MIT|1|
|joblib|1.6.0|BSD-3-Clause|1|
|jsonpatch|1.33|Modified BSD License|1|
|jsonpointer|3.1.1|Modified BSD License|1|
|jsonschema|4.26.0|MIT|1|
|jsonschema-specifications|2025.9.1|MIT|1|
|langchain-core|1.6.6|MIT|1|
|langchain-protocol|0.0.19|MIT|1|
|langgraph|1.2.12|MIT|1|
|langgraph-checkpoint|4.2.0|MIT|1|
|langgraph-prebuilt|1.1.0|MIT|1|
|langgraph-sdk|0.4.5|MIT|1|
|langsmith|0.14.1|MIT|1|
|markdown-it-py|4.2.0|License :: OSI Approved :: MIT License|2|
|MarkupSafe|3.0.3|BSD-3-Clause|1|
|mdurl|0.1.2|License :: OSI Approved :: MIT License|1|
|mpmath|1.3.0|BSD|1|
|narwhals|2.26.0|MIT|1|
|networkx|3.6.1|BSD-3-Clause|1|
|numpy|2.4.6|BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0|19|
|openai|3.22.1|Apache-2.0|2|
|opencv-python-headless|5.0.0.93|Apache 2.0|4|
|opentelemetry-api|1.45.0|Apache-2.0|1|
|orjson|3.12.0|MPL-2.0 AND (Apache-2.0 OR MIT)|3|
|ormsgpack|1.12.2|Apache-2.0 OR MIT|4|
|packaging|26.3|Apache-2.0 OR BSD-2-Clause|3|
|pandas|3.0.6|BSD 3-Clause License|1|
|pillow|12.3.0|MIT-CMU|1|
|pip|26.2.1|MIT|41|
|pluggy|1.6.0|MIT|1|
|portalocker|3.2.0|BSD-3-Clause|1|
|protobuf|7.36.2|3-Clause BSD License|1|
|psycopg|3.3.6|LGPL-3.0-only|1|
|psycopg-binary|3.3.6|LGPL-3.0-only|1|
|pyarrow|25.0.1|Apache-2.0|2|
|pydantic|2.13.5|MIT|1|
|pydantic-settings|2.15.0|MIT|1|
|pydantic_core|2.46.5|MIT|1|
|pydeck|0.9.3|Apache License 2.0|1|
|Pygments|2.21.0|BSD-2-Clause|1|
|pymupdf|1.28.2|Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License|2|
|pytest|9.1.1|MIT|1|
|python-dateutil|2.9.0.post0|Dual License|1|
|python-dotenv|1.2.3|BSD-3-Clause|1|
|python-multipart|0.0.32|Apache-2.0|1|
|pywin32|312|PSF|14|
|PyYAML|6.0.3|MIT|1|
|qdrant-client|1.19.1|Apache-2.0|1|
|referencing|0.37.0|MIT|1|
|regex|2026.9.29|Apache-2.0 AND CNRI-Python|1|
|requests|2.34.2|Apache-2.0|2|
|requests-toolbelt|1.0.0|Apache 2.0|1|
|rich|15.0.0|MIT|1|
|rpds-py|2026.6.3|MIT|1|
|safetensors|0.8.0|License :: OSI Approved :: Apache Software License|1|
|scikit-learn|1.9.1|BSD-3-Clause|4|
|scipy|1.17.1|Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.|5|
|sentence-transformers|6.1.0|Apache-2.0|2|
|setuptools|84.0.0|MIT|17|
|shellingham|1.5.4|ISC License|1|
|six|1.17.0|MIT|1|
|sniffio|1.3.1|MIT OR Apache-2.0|3|
|SQLAlchemy|2.1.1|MIT|1|
|starlette|1.7.0|BSD-3-Clause|1|
|streamlit|1.64.0|Apache-2.0|1|
|sympy|1.14.0|BSD|2|
|tenacity|9.1.4|Apache 2.0|1|
|threadpoolctl|3.7.0|BSD-3-Clause|1|
|tokenizers|0.23.2|License :: OSI Approved :: Apache Software License|1|
|toml|0.10.2|MIT|1|
|torch|2.14.1|Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT|99|
|tqdm|4.70.1|MPL-2.0 AND MIT|1|
|transformers|5.18.0|Apache 2.0 License|1|
|truststore|0.10.4|MIT|1|
|typer|0.27.2|MIT|2|
|typing-inspection|0.4.4|MIT|1|
|typing_extensions|4.16.0|PSF-2.0|1|
|tzdata|2026.4|Apache-2.0|2|
|urllib3|2.8.0|MIT|1|
|uuid_utils|0.17.1|BSD-3-Clause|1|
|uvicorn|0.54.0|BSD-3-Clause|1|
|watchdog|6.0.0|Apache-2.0|2|
|watchfiles|1.3.0|MIT|1|
|websockets|16.1.1|BSD-3-Clause|1|
|xxhash|4.0.1|BSD-2-Clause|1|
|zstandard|0.25.0|BSD-3-Clause|1|
