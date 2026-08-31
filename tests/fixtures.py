"""A tiny synthetic catalog so unit tests run in milliseconds instead of loading 50k rows."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

PRODUCTS = [
    {"parent_asin": "P_LEATHER_BELT", "title": "Classic Leather Belt",
     "features": ["100% Leather", "Buckle closure", "Imported"],
     "description": ["A black leather belt for everyday wear."], "price": 29.99,
     "categories": ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
     "details": {"Department": "Mens"}, "average_rating": 4.5, "rating_number": 9000,
     "store": "Beltworks"},
    {"parent_asin": "P_SUEDE_BELT", "title": "Suede Belt",
     "features": ["Suede", "Buckle closure"], "description": ["A brown suede belt."],
     "price": 45.0, "categories": ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
     "details": {"Department": "Mens"}, "average_rating": 4.1, "rating_number": 40,
     "store": "Suedeco"},
    {"parent_asin": "P_CANVAS_BELT", "title": "Canvas Web Belt",
     "features": ["Cotton", "Slide buckle"], "description": ["A blue canvas belt."],
     "price": 12.0, "categories": ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
     "details": {"Department": "Mens"}, "average_rating": 3.9, "rating_number": 5,
     "store": "Canvasly"},
    {"parent_asin": "P_SILK_SCARF", "title": "Silk Scarf",
     "features": ["Silk", "Hand wash"], "description": ["A red silk scarf."], "price": 60.0,
     "categories": ["Clothing, Shoes & Jewelry", "Women", "Accessories", "Scarves"],
     "details": {"Department": "Womens"}, "average_rating": 4.7, "rating_number": 1200,
     "store": "Scarfery"},
    {"parent_asin": "P_WOOL_SCARF", "title": "Wool Scarf",
     "features": ["Wool", "Machine wash"], "description": ["A grey wool scarf."], "price": 22.0,
     "categories": ["Clothing, Shoes & Jewelry", "Women", "Accessories", "Scarves"],
     "details": {"Department": "Womens"}, "average_rating": 4.2, "rating_number": 300,
     "store": "Woolery"},
]

PROFILE = {"purchase_frequency": "3-4 prior purchases", "average_prior_rating": 4.5,
           "rating_style": "usually positive", "preference_tags": ["fit", "comfort"],
           "summary": "Prior purchases emphasize fit, comfort."}


def write_catalog(directory: str | Path) -> str:
    path = Path(directory) / "catalog.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in PRODUCTS), encoding="utf-8")
    return str(path)


class TempCatalog:
    """Context manager yielding the path to a throwaway catalog file."""

    def __enter__(self) -> str:
        self._dir = tempfile.TemporaryDirectory()
        return write_catalog(self._dir.name)

    def __exit__(self, *exc) -> None:
        self._dir.cleanup()


# ---------------------------------------------------------------------------
# A richer synthetic catalog for the pillar tests.
#
# TempCatalog above is deliberately tiny and several existing tests pin its size,
# so this is a second fixture rather than an extension of the first. It carries
# six distinct category buckets with enough items per bucket to make questions
# about diversity, routing and facet entropy meaningful.
# ---------------------------------------------------------------------------

def _p(pid, title, cats, *, feats=(), desc="", price=None, rating=4.0, n=100,
       store="Store", details=None):
    return {
        "parent_asin": pid, "title": title, "features": list(feats),
        "description": [desc] if desc else [], "price": price,
        "categories": ["Clothing, Shoes & Jewelry", *cats],
        "details": details or {"Department": "Unisex"},
        "average_rating": rating, "rating_number": n, "store": store,
    }


RICH_PRODUCTS = [
    # ---- Accessories Belts -------------------------------------------------
    _p("R_BELT_LEATHER", "Full Grain Leather Belt", ["Men", "Accessories", "Belts"],
       feats=["100% Leather", "Buckle closure"], desc="A black leather belt.",
       price=29.99, n=9000, store="Beltworks"),
    _p("R_BELT_SUEDE", "Suede Belt", ["Men", "Accessories", "Belts"],
       feats=["Suede", "Buckle closure"], desc="A brown suede belt.",
       price=45.0, n=400, store="Suedeco"),
    _p("R_BELT_CANVAS", "Canvas Web Belt", ["Men", "Accessories", "Belts"],
       feats=["Cotton", "Slide buckle"], desc="A blue canvas belt.",
       price=12.0, n=60, store="Canvasly"),
    _p("R_BELT_NYLON", "Nylon Tactical Belt", ["Men", "Accessories", "Belts"],
       feats=["Nylon", "Quick release"], desc="A green nylon belt.",
       price=18.0, n=1500, store="Tactica"),

    # ---- Accessories Scarves -----------------------------------------------
    _p("R_SCARF_SILK", "Silk Scarf", ["Women", "Accessories", "Scarves"],
       feats=["Silk", "Hand wash"], desc="A red silk scarf.",
       price=60.0, n=1200, store="Scarfery"),
    _p("R_SCARF_WOOL", "Wool Scarf", ["Women", "Accessories", "Scarves"],
       feats=["Wool", "Machine wash"], desc="A grey wool scarf.",
       price=22.0, n=300, store="Woolery"),
    _p("R_SCARF_COTTON", "Cotton Scarf", ["Women", "Accessories", "Scarves"],
       feats=["Cotton", "Machine wash"], desc="A white cotton scarf.",
       price=15.0, n=80, store="Scarfery"),

    # ---- Jewelry Earrings --------------------------------------------------
    _p("R_EAR_HOOP", "Stainless Steel Hoop Earrings", ["Women", "Jewelry", "Earrings"],
       feats=["Stainless steel", "Hypoallergenic"], desc="Silver hoop earrings. Fine jewellery for everyday wear.",
       price=19.0, n=7000, store="Hoopsmith"),
    _p("R_EAR_STUD", "Gold Stud Earrings", ["Women", "Jewelry", "Earrings"],
       feats=["Gold plated", "Butterfly back"], desc="Small gold studs. Classic jewellery gift.",
       price=35.0, n=2200, store="Studco"),
    _p("R_EAR_DROP", "Fabric Drop Earrings", ["Women", "Jewelry", "Earrings"],
       feats=["Fabric", "Lightweight"], desc="Pink fabric drop earrings. Handmade jewellery.",
       price=14.0, n=150, store="Spirit Hoops"),

    # ---- Tops & Tees T-Shirts ----------------------------------------------
    _p("R_TEE_COTTON", "Classic Cotton T-Shirt", ["Men", "Tops & Tees", "T-Shirts"],
       feats=["100% Cotton", "Machine Wash"], desc="A white cotton tee. Soft tees for everyday apparel.",
       price=16.0, n=12000, store="Basics Co"),
    _p("R_TEE_POLY", "Performance Tee", ["Men", "Tops & Tees", "T-Shirts"],
       feats=["Polyester", "Moisture wicking"], desc="A black polyester tee. Performance tees and sports apparel.",
       price=24.0, n=3400, store="Sportline"),
    _p("R_TEE_BLEND", "Tri-Blend Tee", ["Men", "Tops & Tees", "T-Shirts"],
       feats=["Cotton", "Polyester", "Rayon"], desc="A grey blend tee. Lightweight tees, casual apparel.",
       price=28.0, n=900, store="Basics Co"),

    # ---- Men Hoodies -------------------------------------------------------
    _p("R_HOOD_COTTON", "Pullover Hoodie", ["Men", "Fashion Hoodies & Sweatshirts"],
       feats=["Cotton", "Kangaroo pocket"], desc="A navy cotton hoodie.",
       price=42.0, n=5600, store="Hoodwork"),
    _p("R_HOOD_FLEECE", "Fleece Zip Hoodie", ["Men", "Fashion Hoodies & Sweatshirts"],
       feats=["Polyester", "Full zip"], desc="A black fleece hoodie.",
       price=38.0, n=1800, store="Warmline"),
    _p("R_HOOD_HEAVY", "Heavyweight Hoodie", ["Men", "Fashion Hoodies & Sweatshirts"],
       feats=["Cotton", "Ribbed cuffs"], desc="A green heavyweight hoodie.",
       price=55.0, n=220, store="Hoodwork"),

    # ---- Shoes Sneakers ----------------------------------------------------
    _p("R_SHOE_RUN", "Running Sneakers", ["Men", "Shoes", "Sneakers"],
       feats=["Mesh upper", "Rubber sole"], desc="Blue running sneakers for the gym. Athletic footwear built for daily miles.",
       price=75.0, n=8800, store="Strideworks"),
    _p("R_SHOE_CANVAS", "Canvas Sneakers", ["Men", "Shoes", "Sneakers"],
       feats=["Cotton canvas", "Lace up"], desc="White canvas sneakers. Casual footwear, trainers for everyday wear.",
       price=40.0, n=2100, store="Canvasly"),
    _p("R_SHOE_LEATHER", "Leather Sneakers", ["Men", "Shoes", "Sneakers"],
       feats=["Leather", "Cushioned insole"], desc="Black leather sneakers. Smart footwear and comfortable trainers.",
       price=95.0, n=640, store="Strideworks"),
]


def write_rich_catalog(directory: str | Path) -> str:
    path = Path(directory) / "rich_catalog.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in RICH_PRODUCTS), encoding="utf-8")
    return str(path)


class RichCatalog:
    """Context manager yielding a path to the six-bucket synthetic catalog."""

    def __enter__(self) -> str:
        self._dir = tempfile.TemporaryDirectory()
        return write_rich_catalog(self._dir.name)

    def __exit__(self, *exc) -> None:
        self._dir.cleanup()


# ---------------------------------------------------------------------------
# The long-tail pathology pair.
#
# analysis/reranker_experiment.json records that the trained model wants a
# popularity-to-phrase ratio of 7.6:1 where the shipped weighted sum uses 1.4:1,
# and analysis/longtail.json measures what that costs: on the 25 sessions whose
# target sits below the 90th popularity percentile of its own shelf the agent
# scores 0.839 against 0.915 elsewhere. Left to optimise this harness the model
# leans on the prior harder than a shopping assistant should.
#
# This is that failure in two products: the thing the shopper described, buried
# under the thing everyone buys. The numbers are calibrated on the real slice --
# the median below-p90 target has about 159 ratings and sits on a shelf whose
# most-reviewed item has about 12,000 -- and the popular one is given every
# advantage the model's other weights reward (a price, more stars, more text) so
# the pair is a harder case than the catalog actually contains.
# ---------------------------------------------------------------------------

PATHOLOGY_PRODUCTS = [
    {"parent_asin": "POPULAR_CANVAS", "title": "Classic Canvas Web Belt",
     "features": ["Cotton canvas", "Slide buckle", "Imported"],
     "description": ["The best selling canvas belt on the site. Durable cotton "
                     "webbing, a slide buckle, and a fit that works with jeans, "
                     "chinos and shorts."],
     "price": 14.99, "categories": ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
     "details": {"Department": "Mens", "Closure": "Buckle"},
     "average_rating": 4.6, "rating_number": 12000, "store": "Canvasly"},
    {"parent_asin": "LONGTAIL_LEATHER", "title": "Full Grain Leather Belt",
     "features": ["100% Leather", "Buckle closure"],
     "description": ["A brown leather belt."], "price": None,
     "categories": ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
     "details": {"Department": "Mens"},
     "average_rating": 4.2, "rating_number": 159, "store": "Smallmaker"},
    {"parent_asin": "MID_NYLON", "title": "Nylon Tactical Belt",
     "features": ["Nylon", "Quick release"],
     "description": ["A green nylon belt."], "price": 18.0,
     "categories": ["Clothing, Shoes & Jewelry", "Men", "Accessories", "Belts"],
     "details": {"Department": "Mens"},
     "average_rating": 4.3, "rating_number": 1500, "store": "Tactica"},
]


class PathologyCatalog:
    """Context manager yielding the popular-canvas / long-tail-leather catalog."""

    def __enter__(self) -> str:
        self._dir = tempfile.TemporaryDirectory()
        path = Path(self._dir.name) / "pathology_catalog.jsonl"
        path.write_text("".join(json.dumps(p) + "\n" for p in PATHOLOGY_PRODUCTS),
                        encoding="utf-8")
        return str(path)

    def __exit__(self, *exc) -> None:
        self._dir.cleanup()
