"""Content identity for an attempt's original question, answer and figures."""
import hashlib
import json


def student_view(value):
    """Do not expose private reference solutions through nested state snapshots."""
    if isinstance(value, list):
        return [student_view(item) for item in value]
    if isinstance(value, dict):
        return {key: student_view({k: v for k, v in item.items() if k != "answer"}
                                  if key == "problem_snapshot" and isinstance(item, dict)
                                  else item)
                for key, item in value.items()}
    return value


def question_snapshot(problem, repository):
    if problem is None:
        return {}, ""
    content = problem.model_dump()
    figures = {}
    for relative in problem.figures:
        path = repository.figure_path(problem.problem_id, relative)
        if path is None:
            raise ValueError("题目配图缺失，不能冻结作答题面")
        figures[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {key: content[key] for key in ("problem_id", "title", "statement", "answer")}
    identity["figure_sha256"] = figures
    version = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                        sort_keys=True).encode()).hexdigest()
    return {**content, "figure_sha256": figures}, version
