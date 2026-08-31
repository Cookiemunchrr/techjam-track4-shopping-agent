"""The V6 hash pins, verified rather than asserted.

The cycle terminated in DECLINE, but three self-referential hashes recorded
inside its artifacts did not match the objects they name (see
analysis/v6_provenance_errata.json). The frozen preregistration artifact was
left byte-intact through the cycle; the only post-termination edit is the
editorial gate rename (D1-D9 -> V6-D1..V6-D9, V7 P2, recorded in the errata),
which does not touch the pinned strings. These tests pin the *true* identities
instead, so the assets and the sealed-validation manifest cannot drift
unnoticed and so the erratum cannot quietly rot.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.shelf_transform import payload_hash  # noqa: E402

ERRATA = ROOT / "analysis" / "v6_provenance_errata.json"
MANIFEST = ROOT / "analysis" / "v6_validation_manifest.json"
SELECTION = ROOT / "analysis" / "v6_selection.json"
ASSETS = {"aliases": ROOT / "assets" / "v6_shelf_aliases.json",
          "mnn": ROOT / "assets" / "v6_shelf_mnn.json"}
ASSET_MANIFESTS = {"aliases": ROOT / "assets" / "v6_shelf_aliases.manifest.json",
                   "mnn": ROOT / "assets" / "v6_shelf_mnn.manifest.json"}

MANIFEST_SHA256 = "05dc809a2f6e1d5fd33659b2334e79fbabfceff9365a6bfc0c60dc23d1b7709e"
PROTOCOL_MANIFEST_PIN = "4dd1c865b875e1bc8a9ab844634d69931f59394dc127ecd76879172363c1669f"
PAYLOADS = {"aliases": "be16b78f7c9a057ed0969e0ca0062b2b5a34135d0554cd028f9c21eba92e4bf6",
            "mnn": "e0905c5bdfb44d876c2648de2134681ac1b21abb237051b6cb7ebe45146e3c7c"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class AssetIdentityTest(unittest.TestCase):
    """Each candidate asset's payload identity, corroborated three ways."""

    def test_payload_hash_recomputes_from_the_committed_mapping(self):
        for mode, path in ASSETS.items():
            with self.subTest(mode=mode):
                self.assertEqual(payload_hash(_load(path)["mapping"]), PAYLOADS[mode])

    def test_asset_manifest_and_selection_agree_with_the_asset(self):
        selection = _load(SELECTION)["candidates"]
        for mode, path in ASSETS.items():
            with self.subTest(mode=mode):
                self.assertEqual(_load(path)["payload_sha256"], PAYLOADS[mode])
                self.assertEqual(_load(ASSET_MANIFESTS[mode])["hashes"]["payload_sha256"],
                                 PAYLOADS[mode])
                self.assertEqual(selection[mode]["payload_sha256"], PAYLOADS[mode])

    def test_alias_pair_count_matches_the_recorded_audit_arithmetic(self):
        """419 pre-removal pairs less 5 removals is the committed 414."""
        manifest = _load(ASSET_MANIFESTS["aliases"])
        counts = manifest["audit"]["counts"]
        mapping = _load(ASSETS["aliases"])["mapping"]
        pairs = sum(len(heads) for heads in mapping.values())
        self.assertEqual(counts["pre_removal"] - counts["removed"], counts["post_removal"])
        self.assertEqual(pairs, counts["post_removal"])
        self.assertEqual(len(manifest["audit"]["removed"]), counts["removed"])


class SealedManifestTest(unittest.TestCase):
    def test_validation_manifest_is_the_object_the_errata_names(self):
        # Line-ending tolerant: the committed blob is LF, but a Windows
        # checkout with core.autocrlf=true materialises CRLF and reads
        # 4dd1c865... -- which is exactly how the protocol's stale pin (E1)
        # was recorded. Normalising here keeps the asserted identity the
        # blob's, from any checkout convention.
        content = MANIFEST.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(content).hexdigest(), MANIFEST_SHA256)

    def test_the_protocols_manifest_pin_is_the_contaminated_value_e1_names(self):
        """E1: the protocol's pin is the CRLF-converted reading of the same
        committed manifest, not the blob's hash. Pin both sides of that."""
        protocol = _load(ROOT / "analysis" / "v6_cycle_protocol.json")
        self.assertIn(PROTOCOL_MANIFEST_PIN, protocol["sealed_validation"]["status"])
        crlf = MANIFEST.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        self.assertEqual(hashlib.sha256(crlf).hexdigest(), PROTOCOL_MANIFEST_PIN)

    def test_sealed_sets_remain_unconsumed(self):
        ledger = _load(ROOT / "analysis" / "v6_validation_ledger.json")
        self.assertEqual(ledger["consumed"], [])


class ErrataTest(unittest.TestCase):
    """The erratum must keep describing a discrepancy that is still real."""

    def test_every_erratum_records_a_hash_that_still_disagrees(self):
        for row in _load(ERRATA)["errata"]:
            with self.subTest(erratum=row["id"]):
                self.assertNotEqual(row["recorded"], row["actual"])

    def test_errata_actuals_are_the_verified_identities(self):
        actual = {row["id"]: row["actual"] for row in _load(ERRATA)["errata"]}
        self.assertEqual(actual["E1"], MANIFEST_SHA256)
        self.assertEqual(actual["E2"], PAYLOADS["aliases"])
        self.assertEqual(actual["E3"], PAYLOADS["mnn"])

    def test_the_stale_pins_are_still_present_where_the_errata_says(self):
        """If a later commit repairs a pin in place, this erratum is obsolete."""
        protocol = _load(ROOT / "analysis" / "v6_cycle_protocol.json")
        audits = _load(ROOT / "analysis" / "v6_blind_mapping_audits.json")
        recorded = {row["id"]: row["recorded"] for row in _load(ERRATA)["errata"]}
        self.assertIn(recorded["E1"], protocol["sealed_validation"]["status"])
        self.assertEqual(
            audits["aliases_reviewed_by_mnn_engineer"]["asset_payload_after_dispositions"],
            recorded["E2"])
        self.assertEqual(audits["mnn_reviewed_by_alias_engineer"]["asset_payload"],
                         recorded["E3"])


class V6D8GateTest(unittest.TestCase):
    """V6-D8 is read from the audit, not hardcoded to pass."""

    def test_gate_reproduces_both_recorded_verdicts(self):
        from tools.v6_compare import v6d8_gate
        aliases, mnn = v6d8_gate("aliases"), v6d8_gate("mnn")
        self.assertTrue(aliases["pass"])
        self.assertFalse(mnn["pass"])
        self.assertAlmostEqual(aliases["precision"], 402 / 407, places=9)
        self.assertAlmostEqual(mnn["precision"], 0.328798, places=6)

    def test_unknown_mode_fails_closed(self):
        from tools.v6_compare import v6d8_gate
        with self.assertRaises(SystemExit):
            v6d8_gate("no_such_candidate")


if __name__ == "__main__":
    unittest.main()
