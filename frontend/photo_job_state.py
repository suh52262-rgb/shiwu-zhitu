"""Bind late photo results to the problem/attempt that created them."""
def submission_rejected(error):
    """A definite client rejection differs from an unknown network outcome."""
    import httpx
    if isinstance(error, httpx.HTTPStatusError):
        code = error.response.status_code
        if 400 <= code < 500 and code != 408:
            return code
    return None


def matches(state, pending, student_id, problem_id):
    if pending.get('student_id') != student_id or pending.get('problem_id') != problem_id:
        return False
    if pending.get('epoch', 0) != state.get('photo_epoch', 0):
        return False
    current = state.get('attempt_id') or ''
    base = pending.get('base_attempt_id') or ''
    return current == base or (not current and pending.get('restored') is True)


def busy(state, student_id, problem_id):
    pending = state.get('photo_pending') or {}
    return matches(state, pending, student_id, problem_id) and pending.get('status') not in {'succeeded','failed','interrupted'}
