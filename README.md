# 🤖 AI 智能体 (AI Agent) 系统

基于 Python + FastAPI + LangChain 构建的完整 AI Agent 系统。支持 ReAct 自主思考循环、工具调用、自我纠错、RAG 长期向量记忆、多轮对话持久化以及 Docker 容器化部署。

## ✨ 核心功能特性

- **🧠 自主思考与 ReAct 循环**：采用 `LangChain create_react_agent` 架构，支持复杂多步任务分解。用户提问后，AI 会"思考→调用工具→观察结果→继续思考"直至完成任务。
- **🔧 丰富的工具扩展库**：内置 `get_weather`（模拟天气）、`get_news`（新闻资讯）、`calculate`（**AST 白名单安全计算器，杜绝任意代码执行风险**）等多个工具。
- **🔍 智能纠错与自我修正**：具备强大的容错机制。例如用户输入"查一下北就天气"，Agent 能自动识别错别字并修正为"北京"，实现自主容错重试。
- **🧩 RAG 长期向量记忆**：集成 `ChromaDB` 向量数据库，支持 `sentence-transformers` 语义嵌入与 `TfidfVectorizer` 自动降级。允许用户注入长期偏好（如"我喜欢吃川菜"），在多轮新对话开启后，Agent 能自动检索并记忆用户习惯，实现个性化推荐。
- **🗣️ 多轮对话与 SQLite 持久化**：支持基于 `Session ID` 的多轮对话上下文记忆。数据存储于 `SQLite` 数据库，配备内存缓存层（30 分钟 TTL），**服务重启后历史聊天记录不丢失**。
- **🖥️ 现代化前端可视化交互**：配套原生 HTML + CSS + JavaScript 前端界面，支持**折叠式思维链（Chain-of-Thought）展示**、错误步骤红色高亮、页面刷新保留记录、一键清空开启新对话，以及多用户会话隔离。
- **🐳 容器化与一键部署**：支持 `Docker` 容器化部署，提供 `Dockerfile` 与数据卷挂载方案；同时提供 Windows 下的 `start.bat` 一键启动脚本，开箱即用。

## 🛠️ 技术栈

| 层级 | 技术 |
|------|------|
| **后端框架** | Python, FastAPI, Uvicorn |
| **AI 框架** | LangChain, LangChain Classic (ReAct Agent), OpenAI / DeepSeek 兼容 API |
| **数据存储** | SQLite (SQLAlchemy ORM), ChromaDB (向量数据库) |
| **嵌入模型** | sentence-transformers (优先) / sklearn TfidfVectorizer (自动降级) |
| **数据规范** | Pydantic v2 (请求/响应模型与接口校验) |
| **前端** | 原生 HTML + CSS + JavaScript (零框架依赖) |
| **部署与运维** | Git, Docker, 一键 Shell/Bat 脚本 |

## 📁 项目结构

```
my-ai-agent/
├── agent.py              # ReAct Agent 核心（LangChain 编排、自我修正、RAG 记忆注入）
├── main.py               # FastAPI Web 服务入口（/chat、/health、/docs）
├── tools.py              # 工具函数库（天气、新闻、安全计算器）
├── memory.py             # RAG 长期记忆模块（ChromaDB + sentence-transformers / Tfidf 降级）
├── database.py           # SQLite 持久化 + 内存缓存层（30min TTL）
├── schemas.py            # Pydantic 数据模型（ChatRequest、ChatResponse、ThoughtStep）
├── config.py             # 全局配置常量
├── index.html            # 前端聊天界面（折叠式思维链、错误高亮、自适应）
├── Dockerfile            # Docker 容器构建文件
├── .dockerignore         # Docker 构建忽略规则
├── .gitignore            # Git 忽略规则
├── requirements.txt      # Python 依赖清单（含版本锁定）
├── start.bat             # Windows 一键启动脚本
└── .env                  # 环境变量（DEEPSEEK_API_KEY）
```

## 🚀 快速体验

### 方式一：一键启动（Windows）

确保已安装 Python 3.10+，双击项目根目录下的 `start.bat`，浏览器自动访问 `http://localhost:8000`。

### 方式二：命令行启动

```bash
cd my-ai-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 方式三：Docker 部署（生产环境推荐）

```bash
# 构建镜像
docker build -t my-ai-agent .

# 运行容器（挂载数据卷以持久化数据库）
docker run -d -p 8000:8000 \
  -e DEEPSEEK_API_KEY=你的密钥 \
  -v ./data:/app \
  --name ai-agent \
  my-ai-agent
```

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat` | 发送消息，返回 AI 回复 + ReAct 思考链 + 长期记忆检索结果 |
| `GET` | `/` | 前端聊天界面 |
| `GET` | `/health` | 健康检查（活跃会话数、版本号） |
| `GET` | `/docs` | Swagger UI 交互式 API 文档 |
| `GET` | `/redoc` | ReDoc API 文档 |

### 请求示例

```json
POST /chat
{
  "message": "查一下北京的天气",
  "session_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### 响应示例

```json
{
  "response": "北京今天天气晴朗，气温22°C，微风，空气质量良好。",
  "status": "success",
  "thoughts": [
    {
      "thought": "用户需要查询北京的天气信息，调用 get_weather 工具",
      "action": "get_weather",
      "action_input": "北京",
      "observation": "【模拟】北京今天晴，22°C，微风，空气质量良",
      "is_error": false
    }
  ]
}
```

## 🧪 功能演示

### ReAct 自主推理

```
用户: 查一下北京的天气，如果空气质量好就推荐户外活动
  💭 思考: 先查北京天气...
  🔧 调用工具: get_weather("北京")
  👁️ 观察: 北京今天晴，22°C，空气质量良
  💭 思考: 空气质量良，适合户外活动，可以给出推荐
  ✅ 回复: 北京今天晴22°C，空气质量良，推荐去公园散步或骑行！
```

### 自我修正

```
用户: 查一下北就的天气
  🔧 调用工具: get_weather("北就")
  👁️ ⚠️ 错误: 未找到城市「北就」，猜测您想查询「北京」
  💭 思考: 上次调用失败，修正参数后重试
  🔧 调用工具: get_weather("北京")
  👁️ 观察: 北京今天晴，22°C
  ✅ 回复: 北京今天晴，22°C
```

### RAG 长期记忆

```
第1轮: 我喜欢吃川菜，住在北京朝阳区
  → [Memory] 提取并存储 2 条长期记忆

（点击"新对话"）

第2轮: 推荐点好吃的
  → [Memory] 检索到: 喜欢吃：川菜, 住在：北京朝阳区
  → Agent 个性化推荐川菜馆
```

## ⚙️ 环境变量

| 变量 | 说明 | 必填 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | ✅ 是 |

## 📄 License

MIT
