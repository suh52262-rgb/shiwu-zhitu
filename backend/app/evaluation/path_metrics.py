"""Explicit route labels and micro-averaged touched-node coverage.

Never select the best matching route from predictions: that would let the model
choose its own ground truth. A method needs an explicit dataset ``methods`` map.
``gold_nodes`` labels a partial answer; ``gold_path`` labels a complete route.
"""
from app.evaluation.metrics import Rate, node_recall


def labeled_node_recall(case: dict, touched: set[str], graph, methods=None) -> Rate:
    methods = methods if isinstance(methods, dict) else {}
    labels = {**case.get("expected", {}), **case}
    label = labels.get("gold_nodes")
    is_path = False
    if label is None:
        label = labels.get("gold_path")
        is_path = label is not None
    if label is None and labels.get("method") is not None:
        method = labels["method"]
        label = methods.get(method) if isinstance(method, str) else None
        is_path = True
    if isinstance(label, str):
        label = methods.get(label)
    if not isinstance(label, list) or not label or any(
        not isinstance(n, str) or not n for n in label
    ):
        return Rate(None, note="缺少有效 gold_nodes/gold_path 或 method 路线映射")
    if graph is None or not set(label) <= graph.node_ids():
        return Rate(None, note="路线标注含未知节点或缺少图，需复核标注")
    if is_path and label not in graph.valid_paths:
        return Rate(None, note="gold_path/method 未对应图中已声明合法路线，需复核标注")
    return node_recall(label, touched)


def aggregate_node_recall(results: list[tuple[str, Rate]]) -> Rate:
    usable = [(cid, r) for cid, r in results if r.computable]
    if not usable:
        return Rate(None, note="没有有效路线节点标注", detail=[
            f"{cid}: n/a（{r.note}）" for cid, r in results
        ])
    numerator = sum(r.numerator for _, r in usable)
    denominator = sum(r.denominator for _, r in usable)
    return Rate(numerator / denominator, numerator, denominator,
                note=f"另有 {len(results) - len(usable)} 条缺少有效标注，已排除"
                if len(usable) < len(results) else "",
                detail=[f"{cid}: {r.render()}" for cid, r in results])
