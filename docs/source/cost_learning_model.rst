===================
Cost Learning Model
===================

The ``Learner`` class calculates edit costs using a probabilistic model. The cost of an edit operation is defined by its "surprisal": a measure of how unlikely the event is based on the training data.

The cost of an event :math:`e` is based on its negative log-likelihood. This value, also known as surprisal, quantifies the amount of information contained in observing that event.

.. math:: \text{cost}(e) = -\log(P(e))

A common, high-probability error will have low surprisal and thus a low cost. A rare, low-probability error will have high surprisal and a high cost.

The model's behavior is determined by the smoothing parameter :math:`k` set via ``with_smoothing()``.


Smoothed Model (k > 0)
----------------------

When :math:`k > 0`, the model uses additive smoothing to provide robust costs that are normalized to the range `[0, 1]`.

Probability for Substitutions and Deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an error :math:`e` (a substitution or deletion) involving a source character :math:`s`, the smoothed conditional probability is:

.. math:: P(e|s) = \frac{c_e + k}{C_s + k \cdot V_s}

- :math:`c_e`: The observed count of the specific error :math:`e`.
- :math:`C_s`: The total number of times the source character :math:`s` appeared.
- :math:`k`: The smoothing parameter.
- :math:`V_s`: The number of possible outcomes for source character :math:`s`. This is the vocabulary size (for all possible substitutions) plus one (for deletion).

Probability for Insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~

For an insertion of a target character :math:`t`, the probability is not conditioned on a source character. Instead, it's conditioned on the entire corpus.

.. math:: P(ins(t)) = \frac{c_{ins(t)} + k}{C_{\text{total}} + k \cdot V_{ins}}

- :math:`c_{ins(t)}`: The observed count of inserting character :math:`t`.
- :math:`C_{\text{total}}`: The total number of all source characters observed in the data.
- :math:`k`: The smoothing parameter.
- :math:`V_{ins}`: The number of possible characters that can be inserted (the vocabulary size).

Bayesian Perspective on Smoothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From a Bayesian point of view, the smoothing parameter :math:`k` acts as the concentration parameter of a **symmetric Dirichlet prior distribution**.

This prior represents a belief about the probability of errors before observing any data. Essentially, we start with a "pseudo-count" of `k` for every possible error. This encodes a prior belief that no error is completely impossible.

- **Prior Belief (k):** A `k` of 1.0 (Laplace smoothing) is equivalent to assuming we have seen every possible error exactly once before looking at the training data.
- **Likelihood (Data):** The observed error counts (e.g., :math:`c_e`) from the training data represent the statistical likelihood.
- **Posterior Probability:** Combines the prior belief with the data's likelihood to produce the posterior probability:

.. math:: P(e|s) = \frac{c_e + k}{C_s + k \cdot V_s} = \frac{\text{Observed Counts } + \text{ Prior Pseudo-Counts}}{\text{Total Observations } + \text{ Total Prior Pseudo-Counts}}

This approach **regularizes** the model, preventing probabilities from becoming zero for unseen events. It helps produce more robust cost estimates, especially with sparse training data, by blending observed frequencies with a baseline assumption of uniformity.

Normalization
~~~~~~~~~~~~~

To ensure costs are capped at a maximum of 1.0, the model calculates a normalization ceiling, :math:`Z`. This ceiling is the maximum possible surprisal for any unseen event. This value is found by identifying the context (i.e., the most frequent source character) that produces the highest cost for a hypothetical unseen error.

Final Cost
~~~~~~~~~~

The final cost is the base surprisal of the event, scaled by the normalization ceiling:

.. math:: w(e) = \frac{-\log(P(e))}{Z}

This ensures all learned costs are in the range `[0, 1]`.

Maximum Likelihood Model (k = 0)
--------------------------------

When :math:`k = 0`, the model performs pure **Maximum Likelihood Estimation (MLE)** without any smoothing or normalization. This is equivalent to using no prior belief in the Bayesian framework, relying entirely on the observed data.

Probability for Substitutions and Deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The probability is the raw observed frequency:

.. math:: P(e|s) = \frac{c_e}{C_s}

Probability for Insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The probability of inserting character :math:`t` is:

.. math:: P(ins(t)) = \frac{c_{ins(t)}}{C_{\text{total}}}

Final Cost
~~~~~~~~~~

The cost is just the surprisal:

.. math:: w(e) = -\log(P(e))

This model has distinct properties:

- **Unbounded Costs:** Costs are not scaled and can be greater than 1.0.
- **Infinite Cost for Unseen Events:** The model assigns an infinite cost to errors not present in the training data (:math:`P=0 \implies \text{cost}=\infty`). In practice, previously unseen edits receive the default costs set on the `WeightedLevenshtein` instance.

A Note on Continuity
--------------------

The cost function is **discontinuous** at :math:`k=0`. As :math:`k` approaches 0 from the right, the normalization ceiling :math:`Z` goes to infinity, causing the limit of the normalized costs to be 0. This does not match the un-normalized MLE calculation.

Therefore, the `k=0` case should be viewed as a separate **Maximum Likelihood Mode** for raw statistical analysis, while `k>0` is the **Smoothing Mode** for practical, normalized distance metrics.
