from ocr_stringdist import weighted_levenshtein_path

print(
    weighted_levenshtein_path(
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
