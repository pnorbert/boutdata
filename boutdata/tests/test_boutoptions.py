from boutdata.data import BoutOptions

import textwrap


def test_getSection_nonexistent():
    options = BoutOptions()
    options.getSection("new")
    assert "new" in options


def test_get_set_item_value():
    options = BoutOptions()
    options["new"] = 5
    assert options["new"] == 5


def test_get_set_item_section():
    options = BoutOptions()
    options["section:new"] = 6
    assert "section" in options
    assert options["section"]["new"] == 6


def test_contains():
    options = BoutOptions()
    options["a:b:c"] = 42

    assert "a" in options
    assert "a:b" in options
    assert "a:b:c" in options
    assert "abc" not in options


def test_as_dict():
    options = BoutOptions()
    options["section:new"] = 7
    expected = {"section": {"new": 7}}
    assert options.as_dict() == expected


def test_rename_section_same_level():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section", "another")

    expected = {
        "top-level value": 0,
        "another": {"first": 1, "second": 2},
        "other top-level": 3,
    }
    assert "another" in options
    assert "section" not in options
    assert options.as_dict() == expected


def test_rename_value_same_level():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section:first", "section:third")

    expected = {
        "top-level value": 0,
        "section": {"third": 1, "second": 2},
        "other top-level": 3,
    }
    assert "section:third" in options
    assert "section:first" not in options
    assert options.as_dict() == expected


def test_rename_value_case_sensitive():
    options = BoutOptions()
    options["lower"] = 0

    options.rename("lower", "LOWER")

    expected = {"LOWER": 0}
    assert options.as_dict() == expected


def test_rename_section_case_sensitive():
    options = BoutOptions()
    options["lower:a"] = 0

    options.rename("lower", "LOWER")

    expected = {"LOWER": {"a": 0}}
    assert options.as_dict() == expected


def test_rename_section_deeper():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section", "another:layer")

    expected = {
        "top-level value": 0,
        "another": {
            "layer": {"first": 1, "second": 2},
        },
        "other top-level": 3,
    }
    assert "another" in options
    assert "section" not in options
    assert options.as_dict() == expected


def test_rename_section_into_other_section():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section1")
    section["first"] = 1
    section["second"] = 2
    section2 = options.getSection("section2")
    section2["third"] = 3
    section2["fourth"] = 4
    options["other top-level"] = 5

    options.rename("section1", "section2")

    expected = {
        "top-level value": 0,
        "section2": {"first": 1, "second": 2, "third": 3, "fourth": 4},
        "other top-level": 5,
    }
    assert options.as_dict() == expected
    assert "section2:third" in options
    assert "section1:first" not in options


def test_rename_value_deeper():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    options.rename("section:first", "section:subsection:first")

    expected = {
        "top-level value": 0,
        "section": {
            "second": 2,
            "subsection": {"first": 1},
        },
        "other top-level": 3,
    }
    assert "section:subsection:first" in options
    assert "section:first" not in options
    assert options.as_dict() == expected


def test_rename_value_into_other_section():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section1")
    section["first"] = 1
    section["second"] = 2
    section2 = options.getSection("section2")
    section2["third"] = 3
    section2["fourth"] = 4
    options["other top-level"] = 5

    options.rename("section1:first", "section2:first")

    expected = {
        "top-level value": 0,
        "section1": {"second": 2},
        "section2": {"first": 1, "third": 3, "fourth": 4},
        "other top-level": 5,
    }
    assert options.as_dict() == expected
    assert "section2:third" in options
    assert "section1:first" not in options


def test_path():
    options = BoutOptions("top level")
    options["a:b:c:d"] = 1
    section = options.getSection("a:b:c")

    assert section.path() == "top level:a:b:c"


def test_str():
    options = BoutOptions()
    options["top-level value"] = 0
    section = options.getSection("section")
    section["first"] = 1
    section["second"] = 2
    options["other top-level"] = 3

    # lstrip to remove the first empty line
    expected = textwrap.dedent(
        """
        top-level value = 0
        other top-level = 3

        [section]
        first = 1
        second = 2
        """
    ).lstrip()

    assert str(options) == expected
