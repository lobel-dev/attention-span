"""Integrity tests for health_config — the single source of truth for tuning values.

These guard the three properties that make the module worth having:

  1. The derived ``ENGINE_DEFAULTS`` mapping stays key-complete and lossless, so the
     ``th=`` override seam every verdict function depends on cannot silently lose a band.
  2. Importing the module is inert - no filesystem, no network - so it can never introduce
     a failure mode into the silent-on-failure render path.
  3. No other module re-declares a band literal. This is the regression guard for the very
     drift the module was created to end: a value must be editable in ONE place.
"""

import dataclasses
import importlib.util
import os
import re
import unittest
from unittest import mock

from attention_span import agent_health, health_config, render
from attention_span import status_catalog as catalog

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "attention_span", "health_config.py")


CONSUMERS = (
    "attention_span/agent_health.py",
    "attention_span/render.py",
    "attention_span/session_ui.py",
    "attention_span/status_catalog.py",
    "attention_span/subagents.py",
    "attention_span/statusline.py",
)


class TestShippedContextBands(unittest.TestCase):
    """The ONE place the shipped context band numbers are spelled out.

    Every other test binds these from the config so that a retune stays a one-file edit.
    That only stays safe if exactly one test notices when the numbers change - this is it.
    An intentional retune updates this class deliberately; an accidental edit fails here.
    """

    def test_shipped_context_bands(self):
        bands = health_config.ENGINE_BANDS
        self.assertEqual(bands.CTX_TIER_STRONG, 32_000)
        self.assertEqual(bands.CTX_TIER_FUNCTIONAL, 128_000)
        self.assertEqual(bands.CTX_TIER_DEGRADING, 200_000)
        self.assertEqual(bands.CTX_TIER_FAILING, 300_000)
        self.assertEqual(bands.CTX_TIER_DEAD, 600_000)

    def test_shipped_200k_class_bands(self):

        bands = health_config.ENGINE_BANDS
        self.assertEqual(bands.CTX_TIER_FUNCTIONAL_200K, 80_000)
        self.assertEqual(bands.CTX_TIER_DEGRADING_200K, 128_000)
        self.assertEqual(bands.CTX_TIER_FAILING_200K, 160_000)
        self.assertEqual(bands.CTX_TIER_DEAD_200K, 192_000)

    def test_the_ladder_descends_strictly_from_worst_to_best(self):

        for cls in (None, *health_config.CONTEXT_CLASS_BAND_OVERRIDES):
            values = [
                getattr(
                    health_config.ENGINE_BANDS,
                    health_config.context_band_field(tier, cls),
                )
                for tier in health_config.CONTEXT_TIER_BANDS
            ]
            self.assertEqual(values, sorted(values, reverse=True), cls)

            self.assertEqual(len(set(values)), len(values), cls)
            self.assertGreater(values[-1], 0, cls)

    def test_every_200k_tier_fires_before_the_wall(self):

        wall = next(
            size for size, cls in health_config.WINDOW_CLASSES.items() if cls == "200k"
        )
        for tier in health_config.CONTEXT_TIER_BANDS:
            field = health_config.context_band_field(tier, "200k")
            self.assertLess(getattr(health_config.ENGINE_BANDS, field), wall, tier)

    def test_there_is_no_window_fill_band_left_to_grade_with(self):

        fields = {f.name for f in dataclasses.fields(health_config.EngineBands)}
        self.assertEqual([name for name in fields if name.startswith("CTX_PCT")], [])


class TestShippedWindowClasses(unittest.TestCase):
    """The ONE place the shipped window-class taxonomy sizes are spelled out.

    Every other test binds sizes from ``health_config.WINDOW_CLASSES``, so teaching the
    taxonomy a new advertised size stays a one-file edit plus this canary.
    """

    def test_shipped_window_classes(self):
        self.assertEqual(
            dict(health_config.WINDOW_CLASSES), {200_000: "200k", 1_000_000: "1m"}
        )

    def test_keys_are_positive_ints_and_values_are_nonempty_strings(self):
        for size, cls in health_config.WINDOW_CLASSES.items():
            self.assertIsInstance(size, int, cls)
            self.assertGreater(size, 0, cls)
            self.assertIsInstance(cls, str, size)
            self.assertTrue(cls, size)

    def test_classifier_resolves_every_class_through_the_map(self):

        for size, cls in health_config.WINDOW_CLASSES.items():
            self.assertEqual(agent_health.window_class(size), cls)


class TestDerivedDefaults(unittest.TestCase):
    """ENGINE_DEFAULTS is derived from the dataclass, never a parallel copy."""

    def test_mapping_covers_every_engine_band_field(self):
        fields = {f.name for f in dataclasses.fields(health_config.EngineBands)}
        self.assertEqual(fields, set(health_config.ENGINE_DEFAULTS))

    def test_derivation_is_lossless_in_value_and_type(self):
        for field in dataclasses.fields(health_config.EngineBands):
            declared = getattr(health_config.ENGINE_BANDS, field.name)
            derived = health_config.ENGINE_DEFAULTS[field.name]
            self.assertEqual(declared, derived, field.name)
            self.assertIs(type(declared), type(derived), field.name)

    def test_engine_defaults_is_the_engine_threshold_source(self):
        self.assertIs(agent_health.DEFAULTS, health_config.ENGINE_DEFAULTS)

    def test_thresholds_resolves_every_band_without_overrides(self):
        resolved = health_config.resolve_thresholds(None)
        self.assertEqual(set(resolved), set(health_config.ENGINE_DEFAULTS))
        self.assertEqual(resolved, dict(health_config.ENGINE_DEFAULTS))

    def test_partial_override_leaves_other_bands_at_their_default(self):
        resolved = health_config.resolve_thresholds({"CTX_TIER_DEGRADING": 120_000})
        self.assertEqual(resolved["CTX_TIER_DEGRADING"], 120_000)
        self.assertEqual(
            resolved["CTX_TIER_FUNCTIONAL"],
            health_config.ENGINE_BANDS.CTX_TIER_FUNCTIONAL,
        )

    def test_every_tier_band_is_reachable_through_the_override_seam(self):

        for field in health_config.CONTEXT_TIER_BANDS.values():
            self.assertIn(field, health_config.ENGINE_DEFAULTS, field)
        for cls, overrides in health_config.CONTEXT_CLASS_BAND_OVERRIDES.items():
            self.assertIn(cls, health_config.WINDOW_CLASSES.values(), cls)
            for tier, field in overrides.items():
                self.assertIn(tier, health_config.CONTEXT_TIER_BANDS, tier)
                self.assertIn(field, health_config.ENGINE_DEFAULTS, field)

    def test_band_field_resolution_prefers_the_class_then_falls_back(self):

        self.assertEqual(
            health_config.context_band_field("degrading", "200k"),
            "CTX_TIER_DEGRADING_200K",
        )
        self.assertEqual(
            health_config.context_band_field("strong", "200k"), "CTX_TIER_STRONG"
        )
        for cls in (None, "1m", "bogus", 7, ["200k"]):
            self.assertEqual(
                health_config.context_band_field("degrading", cls),
                "CTX_TIER_DEGRADING",
                cls,
            )
        self.assertIsNone(health_config.context_band_field("peak", "200k"))
        self.assertIsNone(health_config.context_band_field("bogus", "200k"))

    def test_engine_defaults_cannot_be_mutated_in_place(self):
        with self.assertRaises(TypeError):
            health_config.ENGINE_DEFAULTS["CTX_TIER_DEAD"] = 1

    def test_engine_defaults_helper_returns_an_independent_copy(self):
        copy = health_config.engine_defaults()
        copy["CTX_TIER_DEAD"] = 1
        self.assertNotEqual(health_config.ENGINE_DEFAULTS["CTX_TIER_DEAD"], 1)


class TestImmutability(unittest.TestCase):
    """Frozen config: a stray assignment must fail loudly, not silently retune."""

    def test_engine_bands_are_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            health_config.ENGINE_BANDS.CTX_TIER_DEAD = 1

    def test_presentation_gates_are_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            health_config.PRESENTATION_GATES.DELEGATION_THRESHOLD = 1

    def test_context_tier_maps_cannot_be_mutated(self):
        with self.assertRaises(TypeError):
            health_config.CONTEXT_TIER_BANDS["dead"] = "NOPE"

    def test_window_classes_cannot_be_mutated(self):
        with self.assertRaises(TypeError):
            health_config.WINDOW_CLASSES[500_000] = "500k"

    def test_class_band_overrides_cannot_be_mutated_at_either_level(self):
        with self.assertRaises(TypeError):
            health_config.CONTEXT_CLASS_BAND_OVERRIDES["500k"] = {}
        with self.assertRaises(TypeError):
            health_config.CONTEXT_CLASS_BAND_OVERRIDES["200k"]["dead"] = "NOPE"


class TestStatusCopy(unittest.TestCase):
    """Every tier this module names is a fully worded row in the status catalog."""

    def test_every_tier_has_a_status_word_and_a_style(self):
        for tier in health_config.CONTEXT_TIERS:
            self.assertIn(tier, catalog.STATUSES, tier)
            self.assertTrue(catalog.STATUSES[tier].action, tier)
            self.assertTrue(catalog.STATUSES[tier].color, tier)
            self.assertTrue(catalog.STATUSES[tier].glyph, tier)

    def test_firing_tiers_are_the_four_alerting_tiers(self):

        self.assertEqual(
            set(health_config.CONTEXT_FIRING_TIERS),
            {"dead", "failing", "degrading", "functional"},
        )
        self.assertEqual(
            set(health_config.CONTEXT_TIERS) - set(health_config.CONTEXT_FIRING_TIERS),
            {"peak", "strong"},
        )

    def test_every_tier_colour_resolves_in_the_renderer_palette(self):

        for tier in health_config.CONTEXT_TIERS:
            colour = catalog.STATUSES[tier].color
            self.assertIn(colour, render._RAW, tier)
            self.assertEqual(render.context_tier_color(tier), colour)

    def test_unknown_tiers_have_no_status_copy(self):
        for tier in (None, "", "bogus", "yellow"):
            self.assertEqual(render.context_status(tier), "")

    def test_every_tier_is_its_status_word_alone(self):

        for tier in health_config.CONTEXT_TIERS:
            self.assertEqual(
                render.context_status(tier), catalog.STATUSES[tier].action, tier
            )

    def test_no_tier_wording_carries_a_join_or_a_verb(self):

        for tier in health_config.CONTEXT_TIERS:
            self.assertNotIn("·", render.context_status(tier), tier)

    def test_context_tiers_resolve_through_their_catalog_row(self):

        for tier in health_config.CONTEXT_FIRING_TIERS:
            self.assertEqual(catalog.action_for(tier), render.context_status(tier))

    def test_warming_copy_comes_from_the_catalog(self):
        self.assertEqual(
            catalog.action_for(catalog.WARMING_STATUS), catalog.WARMING_ACTION
        )

    def test_fallback_ladders_never_lengthen(self):

        for status, spec in catalog.STATUSES.items():
            lengths = [len(step) for step in spec.narrow]
            self.assertEqual(lengths, sorted(lengths, reverse=True), status)


class TestImportIsInert(unittest.TestCase):
    """The render path is silent-on-failure; config import must not add I/O."""

    def _load_fresh(self):
        """Import a private copy from source, leaving the cached module alone."""
        spec = importlib.util.spec_from_file_location("_hc_probe", CONFIG_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_import_opens_no_files_and_makes_no_sockets(self):
        with (
            mock.patch("builtins.open", side_effect=AssertionError("opened a file")),
            mock.patch("socket.socket", side_effect=AssertionError("used network")),
        ):
            module = self._load_fresh()
        self.assertEqual(
            dict(module.ENGINE_DEFAULTS), dict(health_config.ENGINE_DEFAULTS)
        )

    def test_config_imports_no_project_modules(self):
        with open(CONFIG_PATH) as handle:
            source = handle.read()
        for name in (
            "agent_health",
            "render",
            "session_ui",
            "statusline",
            "subagents",
            "os",
            "json",
        ):
            self.assertNotRegex(
                source,
                rf"(?m)^\s*(import|from)\s+{name}\b",
                "health_config must stay a literals-only leaf module",
            )


class TestNoBandIsRedeclared(unittest.TestCase):
    """A band literal must live in exactly one file, or 'one-file edit' is a lie."""

    def _band_names(self):
        names = [f.name for f in dataclasses.fields(health_config.EngineBands)]
        names += [f.name for f in dataclasses.fields(health_config.PresentationGates)]
        return names

    def test_consumers_bind_bands_rather_than_restating_literals(self):
        offenders = []
        for filename in CONSUMERS:
            with open(os.path.join(ROOT, filename)) as handle:
                source = handle.read()
            for name in self._band_names():
                for match in re.finditer(
                    rf"(?m)^{re.escape(name)}\s*=\s*(.+)$", source
                ):
                    rhs = match.group(1).split("#", 1)[0]
                    if not any(
                        token in rhs
                        for token in ("_GATES.", "_BANDS.", "health_config.")
                    ):
                        offenders.append(f"{filename}: {name} = {rhs.strip()}")
        self.assertEqual(offenders, [], "re-hardcoded band values found")

    def test_presentation_gate_bindings_match_the_config(self):

        gates = health_config.PRESENTATION_GATES
        for field in dataclasses.fields(health_config.PresentationGates):
            bound = [
                getattr(module, field.name)
                for module in (render, catalog)
                if hasattr(module, field.name)
            ]
            self.assertTrue(bound, "no presentation module binds " + field.name)
            for value in bound:
                self.assertEqual(value, getattr(gates, field.name), field.name)

    def test_no_account_usage_gate_survived_the_layer_it_graded(self):

        names = {f.name for f in dataclasses.fields(health_config.PresentationGates)}
        self.assertEqual(
            sorted(n for n in names if n.startswith(("CAP_", "QUOTA_", "BURNDOWN_"))),
            [],
        )


if __name__ == "__main__":
    unittest.main()
