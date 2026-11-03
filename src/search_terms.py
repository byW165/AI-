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
        # Free words combine with AND; quoted phrases remain one concept.
        tokens = re.findall(r'"([^\"]+)"|(\S+)', value)
        return [(phrase or word,) for phrase, word in tokens]
    groups = []
    remaining = value
    for name in sorted(CONCEPTS, key=len, reverse=True):
        if name in remaining:
            groups.append(CONCEPTS[name])
            remaining = remaining.replace(name, " ")
    remaining = re.sub(r"(?:基于|以及|研究|方法|论文|的|和|与|及)", " ", remaining).strip(" ,，、;；")
    if re.search(r"[\u4e00-\u9fff]", remaining):
        # Unknown terms remain searchable. The UI may translate this original
        # query before requesting arXiv; translation failures keep literal terms.
        groups.append((remaining.strip(),))
        return groups
    if remaining.strip():
        groups.append((remaining.strip(),))
    if not groups:
        return [(value,)]
    return groups


def needs_translation(keyword: str) -> bool:
    """Only concepts without any English alternative need an AI translation."""
    return any(all(re.search(r"[\u4e00-\u9fff]", term) for term in group)
               for group in search_groups(keyword))


def keyword_query(keyword: str) -> str:
    groups = search_groups(keyword)
    expressions = []
    for group in groups:
        terms = [t for t in group if not re.search(r"[\u4e00-\u9fff]", t)] or list(group)
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
    return "；并且 ".join(" / ".join([t for t in group if not re.search(r"[\u4e00-\u9fff]", t)] or list(group)) for group in groups)


