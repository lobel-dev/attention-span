"""Behavior tests for terminal-safe text and compact display helpers."""

import unittest

from attention_span import text


class TestFmtTokens(unittest.TestCase):
    def test_values(self):
        self.assertEqual(text.fmt_tokens(None), "")
        self.assertEqual(text.fmt_tokens(0), "0")
        self.assertEqual(text.fmt_tokens(512), "512")
        self.assertEqual(text.fmt_tokens(2_000), "2K")
        self.assertEqual(text.fmt_tokens(206_070), "206K")
        self.assertEqual(text.fmt_tokens(1_500_000), "1.5M")


class TestSanitize(unittest.TestCase):
    """Control-char stripping at the render seams (ANSI/OSC injection defense)."""

    def test_strips_c0_c1_del_keeps_text(self):
        self.assertEqual(text.sanitize("clean"), "clean")
        self.assertEqual(text.sanitize("a\x1b[31mb"), "a[31mb")
        self.assertEqual(text.sanitize("a\x00\x07\x7f\x9fb"), "ab")
        self.assertEqual(text.sanitize("one\ntwo\ttab"), "onetwotab")

    def test_non_str_is_empty(self):
        self.assertEqual(text.sanitize(None), "")
        self.assertEqual(text.sanitize(123), "")


class TestTokenFormatters(unittest.TestCase):
    """Same numbers, same strings - the formatters only changed address."""

    SAMPLES = (
        (0, "0", "0"),
        (None, "0", "0"),
        ("junk", "0", "0"),
        (-5, "0", "0"),
        (512, "512", "512"),
        (50_000, "50K", "50K"),
        (160_000, "160K", "160K"),
        (2_200_000, "2.2M", "2.2M"),
        (12_900_000, "12.9M", "12.9M"),
        (13_665_260, "13.7M", "13.7M"),
        (999_500_000, "999.5M", "1B"),
        (10**18, "999Q+", "999Q+"),
        (10**30, "999Q+", "999Q+"),
    )

    def test_display_tokens_formats_every_sample(self):
        for value, shown, _ in self.SAMPLES:
            self.assertEqual(text.display_tokens(value), shown, value)

    def test_compact_magnitude_formats_every_sample(self):
        for value, _, compact in self.SAMPLES:
            self.assertEqual(text.compact_magnitude(value), compact, value)

    def test_the_shipped_spot_values(self):
        self.assertEqual(text.display_tokens(50_000), "50K")
        self.assertEqual(text.display_tokens(12_900_000), "12.9M")
        self.assertEqual(text.compact_magnitude(13_665_260), "13.7M")
        self.assertEqual(text.compact_magnitude(10**30), "999Q+")


class TestVisibleWidth(unittest.TestCase):
    """Dingbats follow their real terminal cell width, not a blanket range rule.

    Bare U+2713 / U+26A0 default to text presentation and one cell; a trailing
    U+FE0F flips them to emoji presentation and two. The shipped notice glyph is
    the bare form, so over-counting it skews the shared right edge by a column.
    """

    def test_bare_check_and_warning_are_single_cell(self):
        self.assertEqual(text.vis_len("✓"), 1)
        self.assertEqual(text.vis_len("⚠"), 1)

    def test_east_asian_wide_dingbats_stay_double(self):
        self.assertEqual(text.vis_len("⚽"), 2)
        self.assertEqual(text.vis_len("☔"), 2)

    def test_presentation_selector_widens_a_narrow_base(self):
        self.assertEqual(text.vis_len("✓️"), 2)
        self.assertEqual(text.vis_len("🌗️"), 2)

    def test_emoji_and_frame_glyphs_are_unchanged(self):
        self.assertEqual(text.vis_len("🌗"), 2)
        self.assertEqual(text.vis_len("💀"), 2)
        self.assertEqual(text.vis_len("╭─ ░█ ↺ ·"), 9)


if __name__ == "__main__":
    unittest.main()
