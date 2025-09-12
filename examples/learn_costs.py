from ocr_stringdist import WeightedLevenshtein
from ocr_stringdist.learner import Learner

data = [
    ("kitten", "sitting"),
    ("flaw", "lawn"),
    ("Hallo", "Hello"),
    ("W0rld", "World"),
    ("W0rd", "Word"),
    ("This sentence misses a dot", "This sentence misses a dot."),
    ("This one also does", "This one also does."),
]

learner = Learner().with_smoothing(1.0)

wl = learner.fit(data)
wl = WeightedLevenshtein.learn_from(data)

print("Learned costs:")
print("Substitution costs:")
print(wl.substitution_costs)
print("Insertion costs:")
print(wl.insertion_costs)
print("Deletion costs:")
print(wl.deletion_costs)
