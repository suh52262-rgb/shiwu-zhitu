"""Separate observed writing errors, help operations and unverified legacy labels.

Evidence here is a model judgement with a preserved input, not a validated
diagnosis of the student's understanding. No historical labels are deleted.
"""
POLICY = "paper_evidence_v1"


def evidence_id(attempt_id,evidence):
    import hashlib,json
    basis={key:evidence.get(key) for key in ('concept','issue','student_text','source_id','source_revision','round','node')}
    return hashlib.sha256(json.dumps({'attempt_id':attempt_id,'basis':basis},ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def current_profile_signals(states,legacy=(),student_id=None):
    try:
        from app.services.error_evidence_reviews import decisions
        reviewed=decisions(student_id);status='ready'
    except Exception:
        reviewed=[];status='unavailable'
    result=profile_signals(states,legacy,reviewed)
    result['review_store_status']=status
    if status!='ready':
        result['observed_error_concepts']=[]
        for item in result['error_evidence']:item['review_status']='unavailable'
    return result


def profile_signals(states, legacy=(), decisions=()):
    errors, help_events, historical = [], [], set(legacy or [])
    reviewed={d['evidence_id']:d for d in decisions or []}
    for attempt_id, state in states:
        state = state or {}
        historical.update(state.get("legacy_weak_concepts") or [])
        if state.get("profile_policy_version") != POLICY:
            historical.update(state.get("weak_concepts") or [])
            continue
        for evidence in state.get("error_evidence") or []:
            if (evidence.get("concept") and evidence.get("issue")
                    and evidence.get("student_text")):
                eid=evidence_id(attempt_id,evidence)
                decision=reviewed.get(eid)
                errors.append({**evidence, "attempt_id": attempt_id,'evidence_id':eid,
                    'review_status':decision['decision'] if decision else 'unreviewed',
                    'review_decision':decision})
        help_events.extend({**event, "attempt_id": attempt_id}
                           for event in state.get("help_events") or [])
    return {
        "profile_policy_version": POLICY,
        "observed_error_concepts": sorted({e["concept"] for e in errors if e['review_status'] not in {'dismissed','resolved'}}),
        "error_evidence": errors,
        "help_events": help_events,
        "legacy_weak_concepts": sorted(historical),
        "legacy_status": "unverified_historical_signal",
        "semantic_validated": False,
    }
