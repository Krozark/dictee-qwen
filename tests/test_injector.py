from dictee.injector import APP_WINDOW_TITLE, join_segment, should_inject


def test_join_segment_first_segment_strips_whitespace():
    assert join_segment("", "  bonjour  ") == "bonjour"


def test_join_segment_adds_space_between_words():
    assert join_segment("bonjour", "le monde") == "bonjour le monde"


def test_join_segment_no_space_before_punctuation():
    assert join_segment("bonjour", ", comment ça va") == "bonjour, comment ça va"


def test_join_segment_ignores_empty_new_segment():
    assert join_segment("bonjour", "   ") == "bonjour"


def test_should_inject_true_for_other_window():
    assert should_inject("Mozilla Firefox") is True


def test_should_inject_false_for_app_window():
    assert should_inject(APP_WINDOW_TITLE) is False
