import string

from voice_normal import i18n


def _fields(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_all_languages_have_the_same_keys():
    reference = set(i18n.MESSAGES["en"])
    for lang in i18n.LANGS:
        assert set(i18n.MESSAGES[lang]) == reference, lang


def test_placeholders_match_across_languages():
    for key, english in i18n.MESSAGES["en"].items():
        for lang in i18n.LANGS:
            assert _fields(i18n.MESSAGES[lang][key]) == _fields(english), (lang, key)


def test_unknown_lang_falls_back(monkeypatch):
    monkeypatch.setenv("VOICE_NORMAL_LANG", "de")
    assert i18n.default_lang() == "uk"
