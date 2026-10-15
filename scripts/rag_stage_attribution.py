"""Identity-based stage attribution; outside top-N never means absent from corpus."""


def classify(target, *, indexed_ids, raw_ids, kept_ids, returned_ids,
             loaded_pages, catalogue, delivered, eligible=True):
    identity = target.get('id')
    if not identity:
        return 'unverifiable_identity'
    if not eligible:
        return 'filtered_or_document_unavailable'
    # Positive membership is authoritative even if unrelated records lack IDs.
    matching = [s for s in delivered if s.get('id') == identity]
    if matching:
        if any((s.get('source'), s.get('page')) != (target['doc_id'], target['page']) for s in matching):
            return 'conflicting_delivery_source'
        # Production source_start/end are context offsets, not original-block
        # offsets. Compare actual emitted text; do not misread those coordinates.
        if any(s.get('text') == target['text'].strip() for s in matching):
            return 'delivered_whole_block'
        if any(not s.get('text') or s['text'] not in target['text'] for s in matching):
            return 'unverifiable_delivery_text'
        return 'delivered_partial_block'
    if identity not in indexed_ids:
        return 'not_indexed_in_snapshot'
    if any(s.get('id') == identity for s in catalogue):
        return 'material_selection_or_budget_excluded'
    if any(i is None for i in raw_ids+kept_ids+returned_ids):
        return 'unverifiable_stage_identity'
    if identity in returned_ids:
        return 'returned_hit_not_catalogued'
    if (target['doc_id'], target['page']) in loaded_pages:
        return 'context_loaded_but_expansion_omitted'
    if identity in kept_ids:
        return 'outside_tool_return_top_k'
    if identity in raw_ids:
        return 'outside_kept_top_k_or_deduplicated'
    return 'outside_filtered_vector_top_n'
