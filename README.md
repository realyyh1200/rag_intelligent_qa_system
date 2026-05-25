# RAG智能问答系统

基于FastAPI后端和Vue3前端的AI智能问答平台，支持**自定义RAG构建**和**文件内容总结**。

## 核心功能

- 📚 **自定义RAG构建** - 上传文档，自动构建知识库
- 📝 **文件内容总结** - 自动分析并总结上传文件内容
- 💬 **智能问答** - 基于RAG知识库进行问答
- 🔍 **闲聊检测** - 智能区分闲聊和专业问题
- 📊 **多模态检索** - 支持BM25+向量混合检索
- 🎯 **精排优化** - Cross-encoder精排提升检索精度
- 📱 **流式响应** - 实时流式输出AI回答

## 技术栈

| 层次 | 技术 | 说明 |
|------|------|------|
| 前端 | Vue 3 + Element Plus | 现代化UI框架 |
| 后端 | FastAPI + Python | 高性能API服务 |
| 数据库 | MYSQL | 关系型数据库 |
| 向量存储 | Qdrant | 本地向量数据库 |
| 嵌入模型 | BGE | 中文语义嵌入 |
| 精排模型 | Cross-encoder | 文档相关性精排 |

## 项目结构

```
ai_file_processing_website/
├── backend/
│   ├── api/              # API路由
│   │   ├── auth.py       # 认证相关API
│   │   ├── chat.py       # 聊天相关API
│   │   └── user.py       # 用户相关API
│   ├── core/             # 核心配置
│   │   ├── config.py     # 应用配置
│   │   ├── logger.py     # 日志配置
│   │   └── security.py   # JWT认证和安全工具
│   ├── db/               # 数据库
│   │   └── database.py   # 数据库连接
│   ├── models/           # 数据模型
│   │   └── user.py       # User, Conversation, Message模型
│   ├── schemas/          # Pydantic模式
│   │   └── schema.py     # 请求/响应模式
│   ├── services/         # 业务逻辑
│   │   ├── anthropic_service.py   # AI模型服务
│   │   ├── bge_service.py         # BGE嵌入服务
│   │   ├── cross_encoder_service.py # Cross-encoder精排服务
│   │   ├── memory_service.py      # 记忆服务
│   │   ├── qdrant_service.py      # Qdrant向量服务
│   │   ├── rag_service.py         # RAG检索服务
│   │   └── chit_chat_detector.py  # 闲聊检测服务
│   ├── main.py           # FastAPI应用入口
│   └── requirements.txt  # Python依赖
│
└── frontend/
    ├── src/
    │   ├── api/          # API调用
    │   │   ├── auth.js   # 认证API
    │   │   └── chat.js   # 聊天API
    │   ├── components/   # Vue组件
    │   ├── router/       # 路由配置
    │   ├── store/        # Pinia状态管理
    │   │   └── auth.js   # 认证状态
    │   ├── views/        # 页面视图
    │   │   ├── Login.vue
    │   │   ├── Register.vue
    │   │   └── Chat.vue
    │   ├── App.vue
    │   └── main.js
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- uv（推荐的Python包管理器）
- MYSQL数据库
- Qdrant向量数据库
- BAAI_bge-small-zh BGE嵌入模型
- ms-marco-MiniLM-L-6-v2 Cross-encoder精排模型


### 后端启动

```bash
cd backend

# 安装依赖
uv sync

# 复制环境变量文件并配置
cp .env.example .env
# 编辑.env文件，填入你的API密钥等配置

# 启动服务器
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**API文档地址**: http://localhost:8000/docs

### 前端启动

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

**访问地址**: http://localhost:5173

### 使用批处理脚本（Windows）

```bash
# 启动后端
.\start_backend.bat

# 启动前端
.\start_frontend.bat
```

## RAG工作流程

1. **文档上传** - 用户上传Word/TXT等文本格式文件(PDF等格式还在开发中)
2. **文件解析** - 自动解析文件内容，进行chunk切分
3. **向量嵌入** - 使用BGE模型生成语义向量
4. **向量存储** - 存储到Qdrant向量数据库
5. **智能检索** - 用户提问时进行混合检索（BM25+向量）
6. **精排优化** - 使用Cross-encoder进行相关性精排
7. **智能回答** - 基于检索结果生成精准回答

## 功能特性

- ✅ 用户注册和登录
- ✅ JWT令牌认证（Access Token + Refresh Token）
- ✅ 流式AI响应输出
- ✅ 多对话管理
- ✅ 对话历史持久化
- ✅ 用户信息管理
- ✅ 自定义RAG知识库构建
- ✅ 文件内容自动总结
- ✅ 闲聊/专业问题智能区分
- ✅ 多模态检索（BM25+向量）
- ✅ Cross-encoder精排优化

## 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Anthropic API密钥 | - |
| `SECRET_KEY` | JWT加密密钥 | - |
| `QDRANT_HOST` | Qdrant服务地址 | localhost |
| `QDRANT_PORT` | Qdrant服务端口 | 6333 |

## 使用示例

### 1. 构建知识库

1. 登录系统
2. 上传文档文件
3. 系统自动构建RAG索引

### 2. 智能问答

```
用户：住院电子病历内容有哪些要求？
AI：根据参考文档，住院电子病历包括以下内容...
【参考文献】
- 电子病历怎么填.txt
```

### 3. 文件总结

```
用户：总结一下这份病历文档
AI：这份文档主要讲述了...
```

## 开发说明

### uv 常用命令

| 命令 | 说明 |
|------|------|
| `uv sync` | 同步并安装所有依赖 |
| `uv add package` | 添加新依赖 |
| `uv remove package` | 移除依赖 |
| `uv lock` | 更新锁定文件 |
| `uv run command` | 在虚拟环境中运行命令 |

## 许可证

MIT License