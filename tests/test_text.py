"""Generic dialogue parsing must not depend on the organizer's exact templates."""
from __future__ import annotations

import unittest

from src.text import (is_correction, is_exploring, normalise, split_clauses, tokens)


class NormalisationTest(unittest.TestCase):
    def test_collapses_whitespace_and_lowercases(self):
        self.assertEqual(normalise("  100%\n Leather  "), "100% leather")

    def test_handles_none_and_empty(self):
        self.assertEqual(normalise(None), "")
        self.assertEqual(normalise(""), "")

    def test_nfkc_and_typographic_punctuation_match_ascii(self):
        self.assertEqual(normalise("Women\u2019s \uff24resses\u2014\u201cClassic\u201d"),
                         normalise("Women's Dresses-\"Classic\""))

    def test_tokens_drop_stopwords_and_single_chars(self):
        self.assertEqual(tokens("a the Leather Belt x"), ["leather", "belt"])


class ClauseSplittingTest(unittest.TestCase):
    def test_splits_semicolon_separated_constraints(self):
        self.assertEqual(split_clauses("For that, what matters is: 100% Leather; Buckle closure."),
                         ["100% Leather", "Buckle closure"])

    def test_strips_the_opening_ask_but_keeps_the_category(self):
        self.assertEqual(split_clauses("I'm looking for Accessories Belts. A key requirement is: leather."),
                         ["Accessories Belts", "leather"])

    def test_declined_preference_yields_no_information(self):
        for message in ("I don't have a preference for material; please use your judgment.",
                        "I don't have an additional preference for color.",
                        "No strong opinion on brand."):
            self.assertEqual(split_clauses(message), [], msg=message)

    def test_paraphrased_openings_still_parse(self):
        """The kit warns paraphrasing may be added; these must not fall through."""
        for message in ("I need Accessories Belts", "Show me Accessories Belts",
                        "I'd like Accessories Belts", "I'm after Accessories Belts"):
            self.assertEqual(split_clauses(message), ["Accessories Belts"], msg=message)

    def test_em_dash_clause_boundary_matches_ascii_semicolon(self):
        ascii_form = "I'm looking for Accessories Belts; 100% Leather"
        unicode_form = "I\u2019m looking for Accessories Belts \u2014 100% Leather"
        self.assertEqual(split_clauses(unicode_form), split_clauses(ascii_form))
        self.assertEqual(split_clauses(unicode_form), ["Accessories Belts", "100% Leather"])

    def test_ascii_hyphen_keeps_its_previous_clause_semantics(self):
        self.assertEqual(split_clauses("What matters is: cotton - imported"),
                         ["cotton - imported"])

    def test_curly_apostrophe_decline_is_still_a_decline(self):
        self.assertEqual(split_clauses("I don\u2019t have a preference for material."), [])

    def test_never_returns_empty_or_punctuation_only_clauses(self):
        for clause in split_clauses("Hmm. ... ; , I want leather."):
            self.assertTrue(clause.strip(" ,.;"), "empty clause leaked through")

    def test_tolerates_junk_input(self):
        for message in ("", "   ", "!!!", "\n\t", "🙂🙂", "a" * 5000):
            self.assertIsInstance(split_clauses(message), list)


class DialogueActTest(unittest.TestCase):
    def test_detects_self_correction_generically(self):
        for message in ("Actually, ignore my earlier preference.", "Never mind, I want leather",
                        "Changed my mind - wool instead", "On second thought, silk"):
            self.assertTrue(is_correction(message), msg=message)

    def test_does_not_fire_on_ordinary_turns(self):
        for message in ("I want a leather belt", "What matters is: Buckle closure"):
            self.assertFalse(is_correction(message), msg=message)

    def test_detects_open_ended_browsing(self):
        for message in ("I'm looking for Belts, but I'm still exploring.",
                        "Just browsing for now", "I haven't decided yet"):
            self.assertTrue(is_exploring(message), msg=message)


if __name__ == "__main__":
    unittest.main()
