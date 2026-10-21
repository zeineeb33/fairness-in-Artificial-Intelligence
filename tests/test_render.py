from fairaudit.audit import audit
from fairaudit.render import render_audit_markdown


def test_audit_markdown_sections_and_escaping() -> None:
    report = audit(
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        sensitive={"site": ["a|b", "a|b", "a|b", "c", "c", "c"]},
        y_score=[0.9, 0.1, 0.4, 0.2, 0.8, 0.6],
    )
    md = render_audit_markdown(report, title="Demo")
    assert md.startswith("# Demo")
    assert "## Overall performance" in md
    assert "### `site`" in md
    assert r"a\|b" in md  # pipes in group names must not break the table
    assert "Calibration gap" in md
    assert "(small)" in md
