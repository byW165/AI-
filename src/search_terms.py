"""Transparent bilingual terminology expansion, not fabricated paper content."""
import re

# Each group is a search concept. Broad GNN includes its common model families;
# searching a specific model such as GAT keeps that narrower meaning.
CONCEPTS = {
    "图神经网络": ("graph neural network", "graph convolutional network", "graph attention network", "GNN", "GCN", "GAT", "图神经网络", "图卷积", "图注意力"),
    "图注意力网络": ("graph attention network", "GAT", "图注意力"),
    "图卷积网络": ("graph convolutional network", "GCN", "图卷积"),
    "加速度预测": ("acceleration prediction", "acceleration forecasting", "加速度预测"),
    "车辆预测": ("vehicle prediction", "vehicle trajectory prediction", "vehicle motion prediction", "car-following", "车辆预测"),
    "车速预测": ("vehicle speed prediction", "speed forecasting", "speed prediction", "车速预测"),
    "速度预测": ("speed prediction", "speed forecasting", "速度预测"),
    "车辆轨迹预测": ("vehicle trajectory prediction", "vehicle trajectory forecasting", "车辆轨迹预测"),
    "轨迹预测": ("trajectory prediction", "trajectory forecasting", "motion prediction", "轨迹预测"),
    "跟驰": ("car-following", "car following", "跟驰"),
    "混合动力": ("hybrid electric vehicle", "hybrid vehicle", "混合动力"),
    "能量管理": ("energy management", "能量管理"),
    "物理信息": ("physics-informed", "physics informed", "物理信息"),
    "大模型": ("large language model", "LLM", "大模型"),
}
ALIASES = {"gnn": "图神经网络", "graph neural network": "图神经网络", "graph neural networks": "图神经网络",
           "gat": "图注意力网络", "gcn": "图卷积网络"}


class SearchTermError(ValueError):
    pass


def search_groups(keyword: str) -> list[tuple[str, ...]]:
    value = keyword.strip()
    if not value:
        return []
    if len(value) > 200:
        raise SearchTermError("关键词请控制在 200 字符以内。")
    canonical = ALIASES.get(value.casefold(), value)
    if canonical in CONCEPTS:
        return [CONCEPTS[canonical]]
    if not re.search(r"[\u4e00-\u9fff]", value):
        # Preserve an English scientific phrase as one concept.
        return [(value,)]
    groups = []
    remaining = value
    for name in sorted(CONCEPTS, key=len, reverse=True):
        if name in remaining:
            groups.append(CONCEPTS[name])
            remaining = remaining.replace(name, " ")
    remaining = re.sub(r"(?:基于|以及|研究|方法|论文|的|和|与|及)", " ", remaining).strip(" ,，、;；")
    if re.search(r"[\u4e00-\u9fff]", remaining):
        raise SearchTermError("这个中文词暂未收录术语表，请补充英文关键词。支持如“图神经网络”“图注意力网络”“加速度预测”“轨迹预测”。")
    if remaining.strip():
        groups.append((remaining.strip(),))
    if not groups:
        raise SearchTermError("请输入具体研究术语或英文关键词。")
    return groups


def keyword_query(keyword: str) -> str:
    groups = search_groups(keyword)
    expressions = []
    for group in groups:
        terms = [t for t in group if not re.search(r"[\u4e00-\u9fff]", t)]
        if not terms:
            raise SearchTermError("联网检索需要可识别的英文科研术语。")
        alternatives = []
        for term in terms:
            safe = term.replace('"', " ").replace("\\", " ")
            alternatives.extend([f'ti:"{safe}"', f'abs:"{safe}"'])
        expressions.append("(" + " OR ".join(alternatives) + ")")
    return " AND ".join(expressions)


def expansion_label(keyword: str) -> str:
    groups = search_groups(keyword)
    if not groups:
        return "按所选研究范围检索"
    return "；并且 ".join(" / ".join(t for t in group if not re.search(r"[\u4e00-\u9fff]", t)) for group in groups)


