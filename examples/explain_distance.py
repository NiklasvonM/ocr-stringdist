from ocr_stringdist import explain_weighted_levenshtein

print(
    explain_weighted_levenshtein(
        "Churn Buckets",
        "Chum Bucket",
        substitution_costs={("rn", "m"): 0.5},
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
