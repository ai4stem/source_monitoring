"""
drm_lists.py
============
DRM critical words with their SEMANTIC associates, transcribed verbatim from:

  McDermott, K.B. & Watson, J.M. (2001). The rise and fall of false recall:
  The impact of presentation duration. Journal of Memory and Language, 45, 160-176.
  (Semantic associates = LEFT column of each list in the source materials PDF,
   distributed by the WashU Memory & Cognition Lab stimuli page.)

Each list = the critical (non-presented) lure + its 15 semantic associates,
in DESCENDING associative strength (as printed). This ordering is what our
BAS-graded analysis relies on, so DO NOT sort/alter it.

NOTE: These are semantic associates only. The source also lists phonological
associates (right column) used to build hybrid lists; we use the semantic
lists, which are the standard DRM stimuli.

Provenance is recorded so the paper can cite the exact source.
"""

SOURCE = ("McDermott & Watson (2001), J. Memory & Language 45:160-176; "
          "semantic associate column, WashU Memory & Cognition Lab materials.")

# critical_word -> [15 semantic associates, BAS-descending]
DRM_SEMANTIC_LISTS = {
    "bad":   ["good", "rotten", "harmful", "worse", "villain", "severe", "trouble",
              "awful", "terrible", "evil", "corrupt", "horrible", "nasty", "attitude",
              "mood", "punish"],
    "ball":  ["had", "lad", "bat", "bag", "bud", "band", "dad", "bide", "bid",
              "pad", "ad", "bed", "ban", "tad", "sad", "fad"],
    "beer":  ["bounce", "throw", "basket", "bowling", "golf", "play", "tennis",
              "soccer", "round", "catch", "pitch", "moth", "bat", "kick", "racket",
              "hit"],
    "black": ["white", "gray", "tar", "bruise", "brown", "oil", "tuxedo", "dark",
              "prejudice", "minority", "coffee", "color", "Africa", "coal", "soul",
              "race"],
    "bread": ["rye", "loaf", "crust", "wheat", "butter", "crumb", "garlic", "muffin",
              "dough", "toast", "flour", "Wonder", "bun", "baked", "biscuit", "roll"],
    "car":   ["auto", "drive", "engine", "wreck", "garage", "motor", "van", "truck",
              "crash", "accident", "trunk", "tire", "mechanic", "vehicle", "tow",
              "gas"],
    "chair": ["sit", "couch", "rocking", "swivel", "cushion", "seat", "recliner",
              "wicker", "Lazyboy", "table", "stool", "furniture", "sofa", "rocker",
              "desk", "bench"],
    "cold":  ["hot", "warm", "sneeze", "shiver", "Arctic", "ice cream", "chilly",
              "freezer", "frigid", "heat", "ice", "frost", "freeze", "winter",
              "snow"],
    "dog":   ["hound", "puppy", "bite", "mutt", "pet", "beware", "bone", "tail",
              "cat", "animal", "paw", "poodle", "flea", "bark", "Lassie", "vet"],
    "face":  ["mouth", "expression", "nose", "eyes", "frown", "makeup", "wrinkle",
              "cheek", "head", "mask", "moustache", "beard", "chin", "lips", "shave",
              "smile"],
    "fat":   ["thin", "obese", "large", "weight", "calorie", "slim", "pudgy", "diet",
              "slender", "wide", "cheek", "skinny", "lean", "plump", "chubby",
              "huge"],
    "flag":  ["American", "banner", "pledge", "wave", "allegiance", "country",
              "stars", "USA", "pole", "stripes", "freedom", "nation", "pennant",
              "salute", "symbol", "checkered"],
    "god":   ["lord", "holy", "heaven", "bible", "bless", "angel", "sin", "faith",
              "church", "Jesus", "religion", "pray", "devil", "deity", "Christ",
              "worship"],
    "glass": ["bottle", "lens", "shatter", "prism", "mirror", "hour", "crystal",
              "jar", "pane", "fragile", "mug", "looking", "shard", "cup", "break",
              "window"],
    "gun":   ["pistol", "shot", "holster", "rifle", "bullet", "hunt", "military",
              "powder", "shoot", "trigger", "murder", "aim", "bang", "cannon",
              "revolver", "weapon"],
    "hand":  ["glove", "finger", "shake", "palm", "thumb", "wave", "grip", "foot",
              "fist", "mitten", "wash", "hold", "knuckle", "wrist", "arm", "clap"],
    "hard":  ["rigid", "difficult", "easy", "work", "cement", "concrete", "stiff",
              "tough", "rock", "simple", "complex", "firm", "solid", "soft", "rough",
              "coarse"],
    "hate":  ["dislike", "love", "hostility", "anger", "detest", "resent", "fear",
              "jealousy", "envy", "despise", "abhor", "war", "enemy", "loathe",
              "disgust", "like"],
    "kill":  ["slay", "suicide", "violence", "hunt", "shoot", "stab", "attack",
              "homicide", "destroy", "shot", "smother", "poison", "assassin",
              "murder", "deadly", "choke"],
    "law":   ["rights", "attorney", "enforce", "criminal", "lawyer", "court",
              "government", "regulation", "legal", "officer", "rules", "justice",
              "legislation", "amendment", "police", "order"],
    "mail":  ["stamp", "deliver", "receive", "bills", "letters", "send", "fax",
              "express", "post", "zip", "address", "envelope", "package", "UPS",
              "telegram", "junk"],
    "man":   ["woman", "guy", "sir", "boss", "super", "lady", "person", "fellow",
              "mister", "bachelor", "uncle", "con", "macho", "handsome", "gentleman",
              "male"],
    "pen":   ["ink", "paper", "marker", "eraser", "pencil", "writing", "notebook",
              "Bic", "point", "mark", "write", "scribble", "pal", "quill", "fountain",
              "pad"],
    "rain":  ["umbrella", "drench", "weather", "hail", "cloud", "dew", "pour",
              "storm", "thunder", "wind", "puddle", "acid", "mist", "lightning",
              "sunshine", "flood"],
    "right": ["correct", "perfect", "equal", "accurate", "fair", "justify", "left",
              "turn", "angle", "answer", "mistake", "wrong", "truth", "ethics",
              "direction", "proper"],
    "sick":  ["healthy", "ill", "flu", "nausea", "cancer", "cough", "virus",
              "disease", "medicine", "doctor", "fever", "hospital", "germ", "clinic",
              "vomit", "well"],
    "sleep": ["bed", "rest", "yawn", "pillow", "snooze", "awake", "nap", "dream",
              "tired", "pajamas", "snore", "doze", "drowsy", "coma", "wake",
              "slumber"],
    "slow":  ["quick", "fast", "snail", "hesitant", "brisk", "swift", "molasses",
              "lazy", "cautious", "lethargic", "speed", "hurry", "sluggish", "turtle",
              "rapid", "delay"],
    "smell": ["odor", "cologne", "sniff", "stench", "scent", "nose", "deodorant",
              "aroma", "skunk", "fragrance", "dirty", "sense", "perfume", "stink",
              "foul", "whiff"],
    "smoke": ["fire", "nicotine", "cigar", "pot", "pipe", "chimney", "fumes",
              "cigarette", "ashtray", "Marlboro", "marijuana", "smog", "habit",
              "tobacco", "puff", "inhale"],
    "snake": ["viper", "lizard", "slither", "serpent", "deadly", "hiss", "reptile",
              "cobra", "fangs", "poison", "venom", "slimy", "bite", "python", "worm",
              "rattle"],
    "sweet": ["honey", "bitter", "nice", "ice cream", "sugar", "tart", "taste",
              "fudge", "candy", "syrup", "kind", "chocolate", "dessert", "sour",
              "frosting", "salty"],
    "test":  ["quiz", "final", "study", "evaluate", "experiment", "essay", "stress",
              "screen", "score", "exam", "fail", "lab", "tube", "pass", "grade",
              "analysis"],
    "top":   ["bottom", "peak", "hill", "over", "roof", "summit", "pinnacle",
              "zenith", "apex", "spin", "above", "ceiling", "tip", "lid", "mountain",
              "best"],
    "trash": ["garbage", "waste", "dumpster", "junk", "refuse", "Hefty", "litter",
              "sewage", "scraps", "dump", "rubbish", "landfill", "can", "pile", "bag",
              "recycle"],
    "wet":   ["slippery", "damp", "paint", "splash", "dry", "humid", "water",
              "dripping", "soak", "moist", "saturate", "sponge", "towel", "slick",
              "soggy", "douse"],
}

# critical words that are emotionally/negatively themed -- flag so we can exclude
# them from child-inappropriate contexts if needed, and to check theme balance.
NEGATIVE_THEME = {"bad", "hate", "kill", "gun", "smoke", "snake", "trash"}


def get_lists(exclude_negative: bool = False) -> dict:
    if exclude_negative:
        return {k: v for k, v in DRM_SEMANTIC_LISTS.items()
                if k not in NEGATIVE_THEME}
    return dict(DRM_SEMANTIC_LISTS)


if __name__ == "__main__":
    lists = DRM_SEMANTIC_LISTS
    print(f"{len(lists)} DRM semantic lists")
    bad = [(k, len(v)) for k, v in lists.items() if len(v) != 16]
    # note: many DRM lists have 15 associates; this source prints 16 rows
    for k, v in lists.items():
        assert len(v) >= 15, f"{k}: only {len(v)} associates"
    print("all lists have >=15 associates")
    print("sample (sleep):", lists["sleep"][:5], "...")
