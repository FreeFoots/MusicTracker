# MusicTracker

面向音乐行业从业者的跨平台音乐数据查询平台。输入“我希望获取《空心》的近期数据”，DeepSeek Agent 会调用 QQ 音乐与网易云音乐工具，发现不同歌手的演唱版本，获取可用评论、收藏等数据，返回带来源与采集时间的报告，并将查询和结果自动保存到数据库。

支持多轮追问、作品澄清、查询历史、证据回答和 CSV/JSON 导出。技术栈：Next.js / TypeScript、FastAPI、LangGraph、DeepSeek、SQLAlchemy；数据库支持 PostgreSQL 或本地开发用 SQLite。

当前为本地单用户 v0.6.1。公开入口主要提供当前累计值，不保证完整版本覆盖；真实播放量暂不可用，无历史时不会生成近期增量。

## 快速启动

已验证环境：macOS Tahoe 26.4、Python 3.14、Node.js 26。以下使用 SQLite，无需安装 PostgreSQL。

### 1. 安装依赖

```sh
git clone https://github.com/FreeFoots/MusicTracker.git
cd MusicTracker
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
npm --prefix frontend ci
```

### 2. 配置模型

在项目根目录自行创建本机配置文件 `.env`，填写：

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek_API_Key
```

保存后运行 `chmod 600 .env`。服务地址和模型名使用程序默认值；已有配置请直接编辑，不要覆盖。配置文件仅保存在本机，不要将 Key 提交到 GitHub，也不要放进查询框。未配置 Key 时，可在界面明确选择“规则查询 · 不使用 LLM”。

### 3. 启动与查询

```sh
.venv/bin/python -m musictracker start --sqlite
```

打开 <http://127.0.0.1:4321>，点击“测试 DeepSeek 连接”，选择“DeepSeek Agent”，输入歌曲需求并开始研究。遇到同名作品可选择候选或补充歌手；在会话中继续追问。模型请求会产生 API 费用，查询及公开工具结果会发送到 DeepSeek。

数据保存于 `data/musictracker.db`，再次启动会保留历史。终端保持打开，按 Ctrl+C 停止本次启动的服务；端口被占用时先关闭自己启动的旧实例。

如使用 PostgreSQL，先启动数据库服务，创建专用空库 `createdb musictracker`，再运行 `.venv/bin/python -m musictracker start`（不加 `--sqlite`）。默认使用本机 Unix socket；其他连接方式自行在 `.env` 设置 `DATABASE_URL`。启动时自动执行迁移，不要覆盖已有数据库。
