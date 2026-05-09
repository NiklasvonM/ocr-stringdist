===========================
 Transitive Cost Closure
===========================

By default, :class:`~ocr_stringdist.WeightedLevenshtein` only considers the
costs you explicitly provide. If `("6", "G")` costs `0.5` and deleting `"G"`
costs `0.01`, the engine does **not** automatically know that deleting `"6"`
in context costs `0.51`.

:meth:`~ocr_stringdist.WeightedLevenshtein.transitive_closure` returns a new
instance whose cost dictionaries are filled with these effective (transitive)
costs. The closure also materializes mixed chains, e.g. a `del("y") + ins("x")`
sequence becoming an effective `("y", "x")` substitution.

Example
=======

.. code-block:: python

    from ocr_stringdist import WeightedLevenshtein

    wl = WeightedLevenshtein(
        substitution_costs={("6", "G"): 0.5},
        deletion_costs={"G": 0.01},
    ).transitive_closure()

    # The chain "6" -> "G" -> ε is now a single effective deletion at 0.51.
    print(wl.distance("06", "0"))  # 0.51

After closure, :meth:`~ocr_stringdist.WeightedLevenshtein.explain` returns a
single flat operation at the effective cost; the underlying chain is not
preserved.

You may pass ``prune=True`` to the ``transitive_closure`` method to remove generated substitutions whose costs are already represented by matches, insertions, deletions, or shorter substitutions. This shrinks the resulting cost map but is significantly more expensive to compute.
