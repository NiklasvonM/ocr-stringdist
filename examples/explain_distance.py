from ocr_stringdist import WeightedLevenshtein

print(
    WeightedLevenshtein({("rn", "m"): 0.5}).explain(
        "Churn Buckets",
        "Chum Bucket",
    )
)
# [
#   EditOperation(
#       op_type='substitute',
#       source_token='rn',
#       target_token='m',
#       cost=0.5
#   ),
#   EditOperation(
#       op_type='delete',
#       source_token='s',
#       target_token=None,
#       cost=1.0
#   ),
# ]
