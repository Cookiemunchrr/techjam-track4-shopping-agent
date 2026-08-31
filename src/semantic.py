"""Semantic category resolution — the dense route, without a model.

The pre-rebuild agent matched the customer's category phrase against bucket *names*
by token overlap. That is exact-vocabulary matching wearing a fuzzy coat: it is
invariant to case, word order and articles (measured at +/- 0.000) and it falls off a
cliff the moment the customer uses a different word for the same thing. Synonymising
the head noun took the agent from 0.94365 to 0.64072.

The fix does not need an encoder, because the catalog is already a synonym
dictionary. Shoppers and sellers use the words the bucket name does not:

    "footwear"   2,659 hits, top buckets Shoes Fashion Sneakers / Pumps / Flats
    "timepiece"    382 hits, top bucket  Watches Wrist Watches
    "billfold"      22 hits, top bucket  Card Cases & Money Organizers Wallets
    "shades"       180 hits, top bucket  Sunglasses & Eyewear Accessories Sunglasses
    "jewellery"    149 hits, top buckets Earrings Drop & Dangle / Necklaces

None of those words share a token with the bucket they identify. So instead of
comparing the phrase to bucket names, we learn P(bucket | word) from the words
sellers actually use, and score a phrase by the evidence its words carry.

Two mechanisms, both stdlib:

  vocabulary   an inverted index word -> buckets, mined from product text, pruned
               so generic words ("quality", "great") carry no signal
  morphology   a light suffix stem plus character trigrams, so "sneaker" reaches
               "Sneakers" and near-spellings survive

Deliberately built from the catalog rather than from a generated word list. A
lexicon written by a language model and then tested against synonyms written by the
same model is the closed loop this rebuild exists to remove -- see tools/adversarial.py,
whose vocabulary is disjoint from anything indexed here.
"""
from __future__ import annotations

import collections
import math

from .catalog import Catalog
from .text import tokens

# Pruning. A word seen in almost every bucket identifies none of them.
MIN_DOC_FREQUENCY = 3        # ignore words that appear in fewer products than this
MAX_BUCKET_SPREAD = 0.25     # ignore words spread over more than this share of buckets
BUCKETS_PER_WORD = 12        # keep only the strongest associations per word
TRIGRAM_FLOOR = 0.34         # minimum character-trigram cosine to count as a match


def stem(word: str) -> str:
    """Strip a trailing plural. Crude on purpose -- it must never merge real words."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and not word.endswith(("ses", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def trigrams(text: str) -> set[str]:
    padded = f"  {text.replace(' ', '')}  "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


class Semantic:
    """Word -> bucket evidence, mined once from the frozen catalog."""

    __slots__ = ("catalog", "vocab", "bucket_names", "bucket_grams", "_neighbours")

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.bucket_names = list(catalog.buckets)
        self.bucket_grams = {name: trigrams(name.lower()) for name in self.bucket_names}
        self._neighbours: dict[str, list[tuple[str, float]]] = {}
        self.vocab = self._mine()

    # ------------------------------------------------------------------ build --
    def _mine(self) -> dict[str, dict[str, float]]:
        """Learn P(bucket | word) from the words attached to each bucket's products."""
        bucket_of: dict[str, str] = {}
        for name, members in self.catalog.buckets.items():
            for pid in members:
                bucket_of[pid] = name

        counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        document_frequency: collections.Counter = collections.Counter()
        for pid, frequencies in self.catalog.tf.items():
            bucket = bucket_of.get(pid)
            if bucket is None:
                continue
            for word in frequencies:
                counts[stem(word)][bucket] += 1
                document_frequency[stem(word)] += 1

        total_buckets = max(len(self.bucket_names), 1)
        spread_limit = max(2, int(total_buckets * MAX_BUCKET_SPREAD))

        vocab: dict[str, dict[str, float]] = {}
        for word, per_bucket in counts.items():
            if document_frequency[word] < MIN_DOC_FREQUENCY:
                continue
            if len(per_bucket) > spread_limit:
                continue
            # Weight each association by how concentrated the word is: a word found
            # in two buckets says far more than one found in two hundred.
            concentration = math.log(1.0 + total_buckets / len(per_bucket))
            total = sum(per_bucket.values())
            vocab[word] = {
                bucket: (hits / total) * concentration
                for bucket, hits in per_bucket.most_common(BUCKETS_PER_WORD)
            }
        return vocab

    # ------------------------------------------------------------------ query --
    def resolve(self, phrase: str, limit: int = 5) -> list[tuple[str, float]]:
        """Buckets the phrase could mean, best first."""
        words = [stem(word) for word in tokens(phrase or "")]
        scores: collections.Counter = collections.Counter()

        for word in words:
            for bucket, weight in self.vocab.get(word, {}).items():
                scores[bucket] += weight

        if not scores:
            scores = self._by_trigram(phrase)

        if not scores:
            return []
        best = max(scores.values()) or 1.0
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return [(bucket, score / best) for bucket, score in ranked[:limit]]

    def _by_trigram(self, phrase: str) -> collections.Counter:
        """Last resort: character overlap against bucket names, for odd spellings."""
        wanted = trigrams((phrase or "").lower())
        if not wanted:
            return collections.Counter()
        scores: collections.Counter = collections.Counter()
        for name, have in self.bucket_grams.items():
            shared = len(wanted & have)
            if not shared:
                continue
            cosine = shared / math.sqrt(len(wanted) * len(have))
            if cosine >= TRIGRAM_FLOOR:
                scores[name] = cosine
        return scores

    def neighbours(self, bucket: str, limit: int = 5) -> list[tuple[str, float]]:
        """Buckets that share vocabulary with this one.

        This is what lets the browsing track reach across categories: a customer
        exploring "Running Road Running" should also see walking shoes and athletic
        socks, which share the words their sellers use.
        """
        cached = self._neighbours.get(bucket)
        if cached is not None:
            return cached[:limit]
        if bucket not in self.catalog.buckets:
            return []

        scores: collections.Counter = collections.Counter()
        for word in tokens(bucket):
            for other, weight in self.vocab.get(stem(word), {}).items():
                if other != bucket:
                    scores[other] += weight
        # Reinforce with the vocabulary of the bucket's own members, capped so a
        # huge bucket does not cost more to expand than a small one.
        for pid in self.catalog.buckets[bucket][:40]:
            for word in list(self.catalog.tf[pid])[:24]:
                for other, weight in self.vocab.get(stem(word), {}).items():
                    if other != bucket:
                        scores[other] += weight * 0.05

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:limit * 2]
        best = ranked[0][1] if ranked else 1.0
        result = [(name, score / best) for name, score in ranked]
        self._neighbours[bucket] = result
        return result[:limit]
