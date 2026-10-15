"""Trace a known fact block through saved context and candidate catalogue only."""
def trace_block(block, hits, pool):
    identity = block.get('id')
    if not identity or not block.get('source'):
        return 'unverifiable_identity'
    matching = [s for s in pool if s.get('id') == identity]
    if matching:
        if any(s.get('source') != block['source'] or s.get('page') != block['page'] for s in matching):
            return 'identity_conflict'
        if block.get('indexable') is False or block.get('is_generated'):
            return 'candidate_filter_violation'
        texts = [s.get('text') or '' for s in matching]
        if any(t and t == block.get('text') for t in texts):
            return 'candidate_complete'
        if all(t and (block.get('text') or '').startswith(t) for t in texts):
            return 'candidate_prefix_only'
        return 'source_drift'
    if block.get('indexable') is False or block.get('is_generated'):
        return 'filtered_fact'
    scoped = [h for h in hits if h.get('source') == block['source']]
    if any(identity == s.get('id') for h in scoped for s in h.get('context_blocks', []) + h.get('context_link_candidates', [])):
        return 'saved_context_present_catalogue_excluded'
    if any(h.get('block_id') == identity for h in scoped):
        return 'direct_hit_context_absent'
    return 'absent_from_saved_context'
