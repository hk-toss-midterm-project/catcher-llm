from catcher_llm.utils import format_income_to_10k_won


def test_format_income_to_10k_won_displays_won_amount_in_manwon() -> None:
    """원 단위 연소득을 만원 단위 표시 문자열로 변환하는지 검증한다."""
    assert format_income_to_10k_won(27_200_000) == "2720 만원"
    assert format_income_to_10k_won("72000000") == "7200 만원"


def test_format_income_to_10k_won_keeps_unknown_income_readable() -> None:
    """숫자로 해석할 수 없는 연소득도 화면에서 읽을 수 있게 유지하는지 검증한다."""
    assert format_income_to_10k_won("미입력") == "미입력"
    assert format_income_to_10k_won(None) == "-"
