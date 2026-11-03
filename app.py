"""Run: python -m streamlit run app.py"""
from datetime import timedelta, timezone
import html

import streamlit as st

from src.config import ROOT, ConfigurationError, load_config
from src.fetch_papers import PROFILE_NAMES, fetch_papers
from src.http_client import ServiceError
from src.models import parse_time
from src.ai_jobs import SummaryJob
from src.session_results import SessionResults, filter_records, MAX_SESSION_PAPERS
from src.search_terms import SearchTermError, expansion_label

st.set_page_config(page_title="车辆预测文献追踪", page_icon="📖", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets' / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def show_report(report):
    message = f"AI 摘要处理完成：成功 {report.success} 篇，失败 {report.failed} 篇，复用 {report.skipped} 篇，待处理 {report.remaining} 篇。"
    if report.failed or report.remaining:
        st.warning(message)
    else:
        st.markdown(f'<div class="completion-note" role="status">本批 {report.success} 篇摘要已完成，复用 {report.skipped} 篇。</div>', unsafe_allow_html=True)
    if report.message:
        st.info(report.message)
    if report.errors:
        with st.expander("查看处理错误"):
            for message in report.errors:
                st.text(message)


def cancel_ai():
    job = st.session_state.get("ai_job")
    if job:
        job.cancel()
        st.session_state["ai_stop_requested"] = True


def run_ai(rows, store):
    """Start one session-owned job; never wait for HTTP on the UI thread."""
    old = st.session_state.get("ai_job")
    if old and old.busy:
        st.session_state["ai_notice"] = "上一条 AI 请求仍在收尾，未启动重复调用。论文可继续阅读，请稍后点击生成摘要。"
        return
    if old:
        old.cancel()
    st.session_state.pop("ai_stop_requested", None)
    st.session_state.pop("ai_notice", None)
    st.session_state["ai_job"] = SummaryJob(rows, load_config())


def sync_ai():
    job = st.session_state.get("ai_job")
    if not job:
        return False
    changed = job.tick(st.session_state["paper_results"])
    if not job.active:
        st.session_state["last_report"] = job.report
        st.session_state.pop("ai_job", None)
        st.session_state.pop("ai_stop_requested", None)
        return True
    return changed


def ai_status_panel():
    @st.fragment(run_every=1 if st.session_state.get("ai_job") else None)
    def status():
        if sync_ai():
            st.rerun()
        job = st.session_state.get("ai_job")
        if not job:
            return
        report = job.report
        done = report.success + report.failed
        total = done + report.remaining
        stopped = st.session_state.get("ai_stop_requested", False)
        with st.container(key="ai_activity"):
            if stopped:
                st.info("已停止后续摘要。正在等待当前请求结束；迟到结果会丢弃，论文仍可阅读。")
            else:
                st.progress(done / max(total, 1), text=f"AI 摘要：已处理 {done}/{total} 篇 · 当前请求已等待 {int(job.elapsed)} 秒")
                st.caption(f"正在处理：{job.current_title or '准备请求'}。下方论文已可阅读，结果将逐篇更新。")
                if st.button("停止后续摘要", key="stop_ai"):
                    cancel_ai()
                    st.rerun()
    status()


def reset_filters():
    cancel_ai()
    st.session_state["keyword"] = ""
    st.session_state["code_filter"] = "全部"
    st.session_state.pop("online_search", None)
    st.session_state.pop("paper_results", None)
    st.session_state.pop("last_report", None)


def choose_keyword(value):
    """Fill an example without requesting a source or charging for an AI call."""
    st.session_state["keyword"] = value


def online_request(store, *, keyword, days, profile, limit, auto_ai, previous=None):
    """Explicit actions only; the display and filters never call the APIs."""
    end = parse_time(previous["end"]) if previous else None
    offset = previous["next_start"] if previous else 0
    with st.spinner("正在检索 arXiv，获取论文标题与原摘要……"):
        result = fetch_papers(days, profile, limit, keyword=keyword, offset=offset, end=end)
    if previous is None:
        cancel_ai()
        store.clear()
        st.session_state["code_filter"] = "全部"
    for paper in result.papers:
        store.upsert(paper)
    new_ids = [p.id for p in result.papers]
    all_ids = list(dict.fromkeys((previous["ids"] if previous else []) + new_ids))
    st.session_state["online_search"] = {
        "ids": all_ids, "total": result.total, "truncated": result.truncated,
        "start": result.start.isoformat(), "end": result.end.isoformat(),
        "query": result.query, "keyword": keyword, "profile": profile,
        "days": days, "limit": limit, "next_start": result.next_start,
    }
    if auto_ai and new_ids:
        run_ai([r for r in store.all() if r["id"] in set(new_ids)], store)


def render_paper(row, configured):
    with st.container(key="paper_" + row["id"].replace(".", "_").replace("/", "_")):
        labels = {"Yes": "已开源 · Yes", "No": "未开源 · No", "Unknown": "未说明 · Unknown"}
        code_label = labels.get(row["code_status"], "摘要待生成")
        css = {"Yes": "yes", "No": "no", "Unknown": "unknown"}.get(row["code_status"], "pending")
        date = parse_time(row["published"]).astimezone(timezone(timedelta(hours=8)))
        full_date = date.strftime("%Y-%m-%d %H:%M")
        authors = html.escape("、".join(row["authors"]), quote=True)
        st.markdown(
            f'<div class="paper-top"><span class="paper-meta">arXiv {html.escape(row["id"])}'
            f'<span class="meta-separator">/</span><time title="北京时间首次提交：{full_date}">{date:%Y-%m-%d}</time></span>'
            f'<span class="badge {css}">{code_label}</span></div>'
            f'<h3 class="paper-title">{html.escape(row["title"])}</h3>'
            f'<div class="paper-authors" title="{authors}">{authors}</div>',
            unsafe_allow_html=True,
        )
        if row["status"] == "success":
            st.markdown(
                f'<div class="insight"><div class="field-label">核心贡献</div>'
                f'<div class="insight-text">{html.escape(row["core_contribution"])}</div></div>'
                f'<div class="method-block"><div class="field-label">方法与数据集</div>'
                f'<div class="method-text">{html.escape(row["methods_and_datasets"])}</div></div>',
                unsafe_allow_html=True,
            )
        elif row["status"] == "failed":
            st.warning(row["error"] or "摘要提取失败，可稍后重试。")
        else:
            st.markdown('<p class="pending-note">尚未生成 AI 摘要，可先阅读英文原摘要。</p>', unsafe_allow_html=True)
        with st.expander("英文原摘要"):
            st.write(row["abstract"])
            st.caption("AI 提取依据标题与摘要，研究结论请以原文为准。")
        with st.container(key="paper_actions_" + row["id"].replace(".", "_").replace("/", "_")):
            link_area, retry_area = st.columns([1, 1], vertical_alignment="center")
            link_area.link_button("阅读原文", row["url"], icon=":material/open_in_new:")
            if row["status"] != "success":
                if retry_area.button("生成 / 重试摘要", key=f"retry_{row['id']}", disabled=not configured or bool(st.session_state.get("ai_job"))):
                    run_ai([row], st.session_state["paper_results"])
                    st.rerun()


def main():
    if "paper_results" not in st.session_state:
        st.session_state["paper_results"] = SessionResults()
    store = st.session_state["paper_results"]
    sync_ai()
    config = load_config()
    try:
        config.validate()
        configured = True
    except ConfigurationError:
        configured = False

    with st.sidebar:
        st.markdown('<div class="sidebar-identity"><span class="sidebar-eyebrow">RESEARCH RADAR</span>检索设置<span class="sidebar-subtitle">定义你关心的研究范围</span></div>', unsafe_allow_html=True)
        st.header("检索范围")
        days = st.number_input("最近多少天", min_value=1, max_value=3650, value=180, step=1, key="days",
                               help="按 arXiv 首次提交时间检索。全站只有这一处时间设置。")
        profile = st.selectbox("研究领域", PROFILE_NAMES, key="profile")
        st.caption("综合追踪限定车辆预测；跨领域查找可选“不限领域”。")
        st.divider()
        st.subheader("获取与摘要")
        limit = st.number_input("每批论文数", min_value=1, max_value=50, value=20, key="limit",
                                help="每批 1–50 篇，结果较多时可继续获取下一批。")
        auto_ai = st.checkbox("自动生成 AI 摘要", value=True, key="auto_ai")
        st.caption("关闭后先展示原摘要，可逐篇生成。")
        st.divider()
        if configured:
            st.markdown('<div class="service-state"><span></span>AI 配置已填写</div>', unsafe_allow_html=True)
        else:
            st.warning("填写 AI 配置后可生成结构化摘要，联网检索仍可使用。")
        with st.expander("AI 配置说明", expanded=not configured):
            st.write("在项目 .env 的 AI_API_KEY= 后填写密钥并保存。")
            st.caption("基础地址与模型已预填。配置是否可用以实际调用为准。")
        st.markdown('<div class="session-note"><b>轻量使用，按需阅读</b><br>结果仅在当前页面会话展示。<br>不自动保存，刷新后清空。</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="topbar"><span class="wordmark"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M4 6L19 4L20 19L5 20L12 12L19 4M4 6L12 12L20 19" stroke="currentColor" stroke-width="1.4"/>'
        '<circle cx="4" cy="6" r="2.3" fill="currentColor"/><circle cx="19" cy="4" r="2.3" fill="currentColor"/>'
        '<circle cx="12" cy="12" r="2.3" fill="currentColor"/><circle cx="20" cy="19" r="2.3" fill="currentColor"/>'
        '<circle cx="5" cy="20" r="2.3" fill="currentColor"/></svg>Research Radar<span class="wordmark-divider"></span><span class="wordmark-cn">科研文献工作台</span></span>'
        '<span class="source-label"><i></i>论文来源 arXiv</span></div>'
        '<header class="research-hero"><div class="hero-copy"><span class="hero-eyebrow">让值得读的研究，浮出水面</span>'
        '<h1>车辆预测 · 文献追踪</h1><p>从车辆交互到时序预测，发现新论文，快速读懂核心贡献。</p>'
        '<div class="hero-tags"><span>车辆交互图</span><span>加速度预测</span><span>时空学习</span></div></div>'
        '<div class="hero-art" aria-hidden="true"><svg viewBox="0 0 420 240" fill="none">'
        '<defs><linearGradient id="road-fade" x1="20" y1="20" x2="400" y2="230" gradientUnits="userSpaceOnUse"><stop stop-color="#A4D8C5" stop-opacity=".04"/><stop offset="1" stop-color="#A4D8C5" stop-opacity=".3"/></linearGradient></defs>'
        '<path d="M40 232L163 12M137 240L260 20M238 239L361 19" stroke="url(#road-fade)" stroke-width="40"/>'
        '<path d="M89 232L212 12M187 240L310 20" stroke="#A4D8C5" stroke-opacity=".23" stroke-dasharray="9 12"/>'
        '<path d="M131 180C151 146 180 178 208 132S231 78 260 62" stroke="#B6E9D0" stroke-width="2" stroke-dasharray="4 6"/>'
        '<path d="M207 132L307 116L260 62L150 98L207 132L131 180L307 116" stroke="#B6E9D0" stroke-opacity=".4"/>'
        '<circle cx="207" cy="132" r="43" stroke="#B6E9D0" stroke-opacity=".2"/><circle cx="207" cy="132" r="61" stroke="#B6E9D0" stroke-opacity=".09"/>'
        '<g transform="translate(192 110) rotate(29 15 22)"><rect width="30" height="46" rx="9" fill="#C8EFDE"/><rect x="5" y="9" width="20" height="13" rx="3" fill="#376F65"/><path d="M6 34H24" stroke="#376F65" stroke-width="3" stroke-linecap="round"/></g>'
        '<g transform="translate(246 43) rotate(29 11 17)"><rect width="23" height="36" rx="7" fill="#83B8A5"/><rect x="4" y="7" width="15" height="10" rx="2" fill="#2D5F56"/></g>'
        '<g transform="translate(294 98) rotate(29 11 17)"><rect width="23" height="36" rx="7" fill="#83B8A5"/><rect x="4" y="7" width="15" height="10" rx="2" fill="#2D5F56"/></g>'
        '<circle cx="150" cy="98" r="5" fill="#9DD4BD"/><circle cx="131" cy="180" r="4" fill="#9DD4BD"/>'
        '<path d="M258 60L260 49L270 54" stroke="#B6E9D0" stroke-width="2" stroke-linecap="round"/>'
        '</svg></div></header>',
        unsafe_allow_html=True,
    )

    with st.container(key="search_panel"):
        st.markdown('<div class="search-title-row"><h2 class="search-heading">发现下一篇关键文献</h2><span>学术检索 · AI 结构化摘要</span></div>', unsafe_allow_html=True)
        with st.form("paper_search", border=False):
            field, search_action = st.columns([5, 1.3], vertical_alignment="bottom")
            keyword = field.text_input("论文关键词", placeholder="图神经网络、车辆轨迹预测、NGSIM…", key="keyword",
                                       icon=":material/search:", autocomplete="off")
            submitted = search_action.form_submit_button("联网检索", type="primary", use_container_width=True, key="search_button")
        with st.container(key="suggestions"):
            hint, term1, term2, term3 = st.columns([1.05, 1.3, 1.3, 1.3], vertical_alignment="center")
            hint.markdown('<span class="suggestion-label">试试这些关键词</span>', unsafe_allow_html=True)
            for area, value, key in [(term1, "图神经网络", "quick_gnn"), (term2, "车辆轨迹预测", "quick_trajectory"), (term3, "加速度预测", "quick_acceleration")]:
                area.button(value + " ↗", key=key, on_click=choose_keyword, args=(value,), use_container_width=True,
                            help="填入关键词，再点击联网检索。")
        st.markdown(f'<div class="search-context"><span>当前范围</span>{html.escape(profile)}<span class="context-divider">/</span>最近 {int(days)} 天</div>', unsafe_allow_html=True)

    try:
        expanded = expansion_label(keyword)
        valid_keyword = True
    except SearchTermError as exc:
        st.warning(str(exc))
        expanded = ""
        valid_keyword = False

    if submitted and valid_keyword:
        try:
            online_request(store, keyword=keyword, days=int(days), profile=profile, limit=int(limit), auto_ai=auto_ai)
            st.rerun()
        except (ServiceError, SearchTermError) as exc:
            st.error(f"联网检索失败：{exc} 当前页面已有结果仍保留，请稍后重试。")

    search = st.session_state.get("online_search")
    ai_status_panel()
    if "ai_notice" in st.session_state:
        st.info(st.session_state.pop("ai_notice"))
    if not search:
        st.markdown(
            '<section class="getting-started"><div class="guide-heading"><h2>把检索，变成阅读的起点</h2><span>三步开始你的研究追踪</span></div>'
            '<div class="guide-grid"><article><div class="guide-top"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 7H20M4 17H20M8 4V10M16 14V20" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg><span>01</span></div>'
            '<h3>设定研究范围</h3><p>在左侧选择领域与时间。聚焦细分方向，也可以探索全部 arXiv。</p></article>'
            '<article><div class="guide-top"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5" stroke="currentColor" stroke-width="1.6"/><path d="M16 16L21 21" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg><span>02</span></div>'
            '<h3>找到相关新论文</h3><p>输入中文术语或英文关键词，联网获取论文。结果多时可继续获取下一批。</p></article>'
            '<article><div class="guide-top"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M5 3H15L19 7V21H5V3ZM15 3V7H19M9 11H15M9 15H15M9 18H12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg><span>03</span></div>'
            '<h3>快速读懂核心贡献</h3><p>查看 AI 提取的贡献、方法与开源状态，展开原摘要或前往原文核对。</p></article></div></section>',
            unsafe_allow_html=True,
        )
    else:
        changed = (keyword, int(days), profile, int(limit)) != (search["keyword"], search["days"], search["profile"], search["limit"])
        if changed:
            st.warning("条件已修改，点击“联网检索”更新结果。下方仍显示上次检索的论文。")

        st.markdown(
            f'<div class="results-heading"><h2>检索结果</h2><p>arXiv API 命中 {search["total"]} 篇'
            f'<span class="context-divider">/</span>本轮已获取 {len(search["ids"])} 篇</p></div>',
            unsafe_allow_html=True,
        )
        st.caption(f'本轮检索：{search["keyword"] or "领域预设"}；{search["profile"]}，最近 {search["days"]} 天。')
        with st.container(key="results_toolbar"):
            status_filter, reset_action = st.columns([4, 1.3], vertical_alignment="bottom")
            code = status_filter.selectbox("代码开源状态", ["全部", "Yes", "No", "Unknown"],
                                            format_func=lambda x: {"全部": "全部状态", "Yes": "已开源 · Yes", "No": "明确未开源 · No", "Unknown": "未说明 · Unknown"}[x],
                                            key="code_filter", help="只按本轮摘要判定：Yes 为明确公开，No 为明确不公开，Unknown 为未说明。未生成摘要时尚无代码状态。")
            reset_action.button("清空结果", on_click=reset_filters, key="reset_filters", type="tertiary", use_container_width=True)

        visible = filter_records(store.all(), code)
        with st.container(key="metrics"):
            a, b, c, d = st.columns(4)
            a.metric("当前显示", len(visible))
            b.metric("显示中 · 摘要完成", sum(r["status"] == "success" for r in visible))
            c.metric("显示中 · 已开源", sum(r["code_status"] == "Yes" for r in visible))
            d.metric("显示中 · 待处理", sum(r["status"] != "success" for r in visible))
        if "last_report" in st.session_state:
            show_report(st.session_state.pop("last_report"))

        pending = [r for r in visible if r["status"] != "success"]
        if pending:
            if st.button("生成 / 重试当前匹配的摘要", disabled=not configured or bool(st.session_state.get("ai_job")), key="retry_all",
                         help=f"每次最多处理 {int(limit)} 篇，本轮成功摘要会直接复用。"):
                run_ai(pending[:int(limit)], store)
                st.rerun()
        if not visible:
            if search["total"] == 0:
                st.info("本次检索未找到匹配论文。可扩大左侧时间范围、调整关键词，或改选“不限领域”。")
            else:
                st.info("本批结果中没有符合该开源状态的论文。可选“全部状态”查看；尚未生成摘要的论文还没有开源判定。")

        pages = max(1, (len(visible) + 9) // 10)
        page = int(st.selectbox("结果页码", range(1, pages + 1), key=f"page_{pages}")) if pages > 1 else 1
        for row in visible[(page - 1) * 10:page * 10]:
            render_paper(row, configured)

        if search["truncated"]:
            at_capacity = len(search["ids"]) + search["limit"] > MAX_SESSION_PAPERS
            with st.container(key="load_more_area"):
                st.caption(f'当前已取得 {len(search["ids"])} / {search["total"]} 篇，按首次提交时间从新到旧排列。')
                if st.button(f"继续联网获取下一批（最多 {search['limit']} 篇）", key="load_more", disabled=changed or at_capacity or bool(st.session_state.get("ai_job")),
                             icon=":material/expand_more:", use_container_width=True):
                    try:
                        online_request(store, keyword=search["keyword"], days=search["days"], profile=search["profile"],
                                       limit=search["limit"], auto_ai=auto_ai, previous=search)
                        st.rerun()
                    except ServiceError as exc:
                        st.error(f"获取下一批失败：{exc} 当前结果仍保留。")
                if at_capacity:
                    st.caption("当前会话最多展示 200 篇，可清空后重新检索。")
        with st.expander("检索详情与术语展开"):
            st.text("实际检索词：" + expansion_label(search["keyword"]))
            st.text(f'首次提交时间 UTC：{search["start"]} 至 {search["end"]}')
            st.code(search["query"], language="text")
            st.caption("API 命中表示符合索引查询条件，论文相关性与 AI 摘要请结合原文核对。")

    st.markdown('<footer class="footer"><span>车辆预测文献追踪</span><span>原摘要可核查 · 结果不自动保存</span></footer>', unsafe_allow_html=True)


try:
    main()
except OSError:
    st.error("配置或页面样式文件无法访问，请检查项目目录权限。")
