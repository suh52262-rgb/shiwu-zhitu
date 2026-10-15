from app.services.hint_citations import render_hint

POOL=[{'id':'H0','source':'student_answer','text':'任何入射角、任意厚度透镜都可直接用公式。'}]


def test_server_copies_exact_wording_and_keeps_offsets():
    text,refs,bad=render_hint('再核对⟦H0:任何入射角⟧的适用条件。',POOL)
    assert not bad
    assert text=='再核对任何入射角的适用条件。'
    assert refs[0]['text']==POOL[0]['text'][refs[0]['start']:refs[0]['end']]
    assert not refs[0]['semantic_validated']


def test_synonym_cannot_replace_literal_quote_or_hide_in_free_quotes():
    _,_,bad=render_hint('再核对⟦H0:任意入射角⟧。',POOL)
    assert bad
    assert render_hint("再核对'任意入射角'。",POOL)[2]


def test_nonexistent_source_and_out_of_range_are_rejected():
    assert render_hint('⟦H9:0-3⟧',POOL)[2]
    assert render_hint('⟦H0:0-999⟧',POOL)[2]


def test_variable_prime_is_not_misread_as_quotation():
    assert not render_hint("比较x'与y'的含义。",POOL)[2]
