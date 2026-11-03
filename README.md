# 车辆预测 AI 文献追踪看板

第三周课程作业：围绕**车辆加速度预测、车辆交互图与时序方法**，检索相关车速、跟驰、轨迹预测及混动车能量管理论文，并用 AI 提取结构化摘要。

Python 3.12 + Streamlit + arXiv API + DeepSeek / 兼容接口。论文仅在当前页面会话中展示，**不自动保存、不建立本地论文库、不自动下载 PDF**。

源码仓库：[byW165/AI-](https://github.com/byW165/AI-)。应用在本机运行，上传源码不等于部署公网网站。

## 作业要求与实现

| 要求 | 实现 |
| --- | --- |
| 自动数据检索 | 提交关键词后，程序自动请求 arXiv；最近 X 天可设 1–3650，支持分批获取 |
| AI 结构化提取 | 三组人工 Few-shot 示例 + JSON；提取核心贡献、方法/数据集、代码状态 |
| 交互看板 | 中文常见术语与英文关键词搜索、开源状态筛选、原摘要、原文链接 |
| 防幻觉与错误兜底 | 仅依据标题与摘要；网络、限流、JSON/字段校验错误有超时及有限重试 |
| PRD → TDD → Execution | [研究说明.md](研究说明.md)、[技术设计文档.md](技术设计文档.md)、[progress.md](progress.md) |

自动检索指用户提交后自动拉取和解析，不包含无人值守定时任务。代码状态保留 Yes/No，并增加 Unknown：未提及或仅承诺将来公开时不猜成 No。

## 安装与启动

已验证环境：Windows、Python 3.12.14。建议使用 Python 3.12。首次在 PowerShell 执行：

```powershell
git clone https://github.com/byW165/AI-.git
cd AI-
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次将 `.env.example` 复制为 `.env`，填写自己的密钥；已有 `.env` 时直接编辑，不要覆盖。

```dotenv
AI_BASE_URL=https://api.deepseek.com
AI_API_KEY=在本地填写自己的密钥
AI_MODEL=deepseek-flash
```

基础地址和模型可配置，模型名称以 [DeepSeek 官方文档](https://api-docs.deepseek.com/)为准。密钥只放本地 `.env` 或环境变量，不提交 GitHub。环境变量优先；保存配置后重新点击生成/重试即可加载。

Windows 双击 **启动看板.bat**。启动器会复用已有服务，避让占用端口，并在服务就绪后打开浏览器；启动窗口关闭不影响后台服务。使用启动器打开的实际地址，不必记端口。

也可以手动运行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

手动运行默认打开 http://127.0.0.1:8501 ，保持终端运行，Ctrl+C 停止。macOS/Linux 可使用 `python3 -m venv .venv` 和 `.venv/bin/python` 执行相同命令；未做这些平台的实机验收。

## 怎么使用

1. 左侧设置最近多少天、研究领域和每批论文数（默认 20，最多 50）。
2. 输入“图神经网络”、GAT、车辆轨迹预测或数据集关键词，点击**联网检索**或按 Enter。下方示例词点击只填入，不自动发请求。
3. 来源返回即显示论文。勾选自动 AI 时，摘要在后台逐篇更新；页面显示当前论文、等待秒数和进度。
4. 按代码状态筛选；展开英文原摘要、打开原文核对。命中较多时主动获取下一批，当前会话最多 200 篇，每页显示 10 篇。
5. 服务繁忙可先阅读原摘要，稍后单篇或批量重试。**停止后续摘要**不再调度下一篇；当前请求可能仍需收尾，迟到结果会丢弃。

默认“综合追踪”限定车辆预测。如果想查全领域图神经网络，选择“不限领域（全部 arXiv）”。来源是 arXiv，不覆盖所有期刊会议或全网网站。

中文常见术语会透明展开为英文表达式，检索详情可核对。未收录中文会提示补充英文；这不是任意中文语义搜索。结果相关性仍需人工判断。

全站只有左侧一处时间设置，按 UTC 首次提交日期筛选；卡片显示北京时间。修改检索条件后重新提交才更新结果。状态筛选、详情和分页不会再次调用 AI。

新搜索成功替换旧结果；清空、刷新或会话结束后不恢复论文与摘要。需要文件时自行前往原文下载，下次生成摘要可能重新消耗 API 额度。

## AI 状态与限制

- Yes：摘要明确已公开代码；No：明确不公开；Unknown：未说明或只是未来发布。
- AI 不补写未出现的数据集、代码链接或能耗效果；三个示例是输出规范，不是展示用论文。
- 同一会话同时只进行一个 AI 请求。服务故障耗尽预算会暂停后续任务；JSON 格式失败按单篇处理。
- HTTP 503 是服务繁忙，401 是认证失败，402 是余额不足。“配置已填写”仅表示字段完整，不保证实时在线。见 [DeepSeek 错误码](https://api-docs.deepseek.com/quick_start/error_codes/)。
- 仅凭标题与摘要无法确认全文中的全部实验与开源信息，研究结论需核对原文。
- 本次交付验收：真实中文搜索取得论文；DeepSeek 本轮返回 503，先前同一配置已有成功提取记录。外部服务波动不伪装成成功，详见进度日志。

## 验证

离线测试不调用真实 AI：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

真实检索（只打印，不保存论文）：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.verify_live --days 180 --limit 3 --keyword "图神经网络"
```

如需验 AI，追加 `--ai`，只对首篇提取，可能消耗额度。模拟异常测试与真实接口结果分别记录，不能相互替代。

## 文件结构

```text
AI-/
├── AGENTS.md             # 开发规则与 AI 协作边界
├── 研究说明.md            # Milestone 1：PRD
├── 技术设计文档.md         # Milestone 2：架构、数据流、API、界面设计
├── progress.md           # Milestone 3：验收与迭代日志
├── README.md             # 安装、使用、验证
├── app.py                # 看板入口
├── requirements.txt      # 已验证的直接及传递依赖
├── 启动看板.bat           # Windows 启动入口
├── .env.example          # 不含密钥的配置模板
├── .gitignore
├── .gitattributes
├── .streamlit/config.toml
├── src/                  # 来源、提取、后台任务、会话结果
├── assets/style.css
├── scripts/              # 启动器和真实接口检查
└── tests/                # 离线测试与 AppTest
```

本机的 `.env`、`.venv/` 和 `data/` 不上传：分别用于密钥、运行环境、启动器状态/诊断日志。data 不包含论文或 AI 结果。Python 缓存和数据库文件也被 Git 忽略。
