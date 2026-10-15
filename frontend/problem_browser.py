"""Search published metadata without reading answers or changing attempts."""
import unicodedata


def normalize(value):
    return unicodedata.normalize('NFKC', str(value or '')).strip().casefold()


def problem_tags(problem):
    values = problem.get('knowledge_points') or []
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(v.strip() for v in values if isinstance(v, str) and v.strip()))


def tag_choices(problems):
    return sorted({tag for p in problems.values() for tag in problem_tags(p)})


def matching_ids(problems, query='', tags=(), *, require_all=False):
    selected = {normalize(t) for t in tags}
    words = normalize(query).split()
    result = []
    for pid, problem in problems.items():
        labels = problem_tags(problem)
        label_keys = {normalize(t) for t in labels}
        if selected and not (selected <= label_keys if require_all else selected & label_keys):
            continue
        haystack = normalize(' '.join([pid, str(problem.get('title') or ''), *labels]))
        if all(word in haystack for word in words):
            result.append(pid)
    return result


def selection_options(matches, current):
    """Keep the current problem even when filters exclude it; switching is explicit."""
    return list(matches) if current in matches else [current, *matches]
