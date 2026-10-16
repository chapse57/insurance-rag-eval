from rag.view import display_quote, lead_in, lead_out, with_context, guard_label, highlight


def test_highlight_marks_quote_even_with_different_spacing():
    text = "중도해지하는 경우 해지환급금이 없습니다. 다만, 월납이 아닌 경우"
    out = highlight(text, "해지환급금이없습니다")
    assert "<mark>해지환급금이 없습니다</mark>" in out


def test_highlight_escapes_html_and_ignores_missing_quote():
    out = highlight("<b>약관</b> 내용", "전혀 없는 문장입니다")
    assert "<mark>" not in out and "&lt;b&gt;" in out


def test_guard_label_known_and_unknown():
    assert guard_label("ok")[0] == "통과"
    assert guard_label("quote_not_found")[0] == "차단"
    assert guard_label("weird")[0] == "weird"


def test_display_quote_trims_boundary_fragment():
    assert display_quote("써, 중도해지하는 경우 환급금은 없습니다") == "…중도해지하는 경우 환급금은 없습니다"


def test_display_quote_trims_mid_word_start():
    chunk = "보험계약을 해지하는 경우 환급금은 없습니다."
    assert display_quote("약을 해지하는 경우 환급금은 없습니다", chunk) == "…해지하는 경우 환급금은 없습니다"


def test_display_quote_keeps_clean_quote():
    chunk = "회사는 보험계약을 해지하는 경우 환급금은 없습니다."
    assert display_quote("보험계약을 해지하는 경우", chunk) == "보험계약을 해지하는 경우"


def test_lead_in_takes_last_sentence_of_prev_chunk():
    assert lead_in("가입합니다. 보험료를 연체하면 보험금을 지급하지 않습니다. 해지환급금은 납입한 보험료에서 빼") == "해지환급금은 납입한 보험료에서 빼"


def test_lead_out_stops_at_first_sentence_end():
    assert lead_out("계산합니다. 다음 조항은 따로 봅니다.") == "계산합니다."


def test_with_context_keeps_quote_mark_and_greys_neighbors():
    out = with_context("써, 중도해지하는 경우 없습니다.", "중도해지하는 경우 없습니다", "보험료를 빼고 남은 돈을 돌려주지 않아", "")
    assert out.startswith('<span style="opacity:0.45">보험료를 빼고 남은 돈을 돌려주지 않아</span>')
    assert "<mark>" in out
