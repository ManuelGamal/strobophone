import pytest
from strobophone.voice import parse_quantity, verified_parameters, words_to_number


@pytest.mark.parametrize(
    "text, value",
    [
        ("eighty", 80),
        ("one hundred", 100),
        ("two hundred and forty", 240),
        ("one point two six", 1.26),
        ("twelve hundred", 1200),
        ("zero point five", 0.5),
        ("1.26", 1.26),
        ("three thousand five hundred", 3500),
    ],
)
def test_words_to_number(text, value):
    assert words_to_number(text) == pytest.approx(value)


@pytest.mark.parametrize(
    "quote, value, unit",
    [
        ("eighty centimetres", 0.8, "m"),
        ("one point two six metres", 1.26, "m"),
        ("fifty grams", 0.05, "kg"),
        ("one hundred hertz", 100, "Hz"),
        ("two hundred and forty frames per second", 240, "fps"),
        ("40 mm", 0.04, "m"),
        ("1.5 kilograms", 1.5, "kg"),
    ],
)
def test_parse_quantity_to_si(quote, value, unit):
    v, u = parse_quantity(quote)
    assert v == pytest.approx(value) and u == unit


def test_quantity_without_unit_is_rejected():
    with pytest.raises(ValueError):
        parse_quantity("eighty")


def test_only_quotes_found_verbatim_in_the_transcript_are_accepted():
    transcript = "String length eighty centimetres. Hanging mass fifty grams."
    found = [
        {"parameter": "string_length", "quote": "eighty centimetres"},
        {"parameter": "hanging_mass", "quote": "sixty grams"},  # not what was said
        {"parameter": "drive_frequency", "quote": "one hundred hertz"},  # never said
    ]
    params, rejected = verified_parameters(transcript, found)
    assert params == {"string_length": (pytest.approx(0.8), "m")}
    assert [r["parameter"] for r in rejected] == ["hanging_mass", "drive_frequency"]


def test_unit_must_suit_the_parameter():
    params, rejected = verified_parameters(
        "drop height fifty grams",
        [{"parameter": "drop_height", "quote": "fifty grams"}],
    )
    assert params == {} and rejected[0]["reason"].startswith("unit")
