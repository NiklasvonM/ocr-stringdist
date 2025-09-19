=====================
 Cost Learning Model
=====================

The ``Learner`` class calculates edit costs using a probabilistic model. The cost of an edit operation is defined by its **surprisal**: a measure of how unlikely that event is based on the training data. This value, derived from the negative log-likelihood :math:`-\log(P(e))`, quantifies the information contained in observing an event :math:`e`.

A common, high-probability error will have low surprisal and thus a low cost. A rare, low-probability error will have high surprisal and a high cost.

-------------------
Probabilistic Model
-------------------

The model estimates the probability of edit operations and transforms them into normalized, comparable costs. The smoothing parameter :math:`k` (set via ``with_smoothing()``) allows for a continuous transition between a Maximum Likelihood Estimation and a smoothed Bayesian model.

General Notation
~~~~~~~~~~~~~~~~

- :math:`c(e)`: The observed count of a specific event :math:`e`. For example, :math:`c(s \to t)` is the count of source character :math:`s` being substituted by target character :math:`t`.
- :math:`C(x)`: The total count related to a context :math:`x`. For example, :math:`C(s)` is the total number of times the source character :math:`s` appeared.
- :math:`V`: The total number of unique characters in the vocabulary.

Probability of Substitutions and Deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a substitution :math:`s \to t` or a deletion of :math:`s`, the smoothed conditional probability is:

.. math:: P(e|s) = \frac{c(e) + k}{C(s) + k \cdot (V+1)}

Here, the error space for a source character consists of substitutions to any of the :math:`V` vocabulary characters, plus one outcome for deletion, resulting in :math:`V+1` total possibilities.

Probability of Insertions
~~~~~~~~~~~~~~~~~~~~~~~~~

For an insertion of a target character :math:`t`, the probability is conditioned on the entire corpus:

.. math:: P(\text{ins}(t)) = \frac{c(\text{ins}(t)) + k}{C_{\text{total}} + k \cdot V}

Here, :math:`C_{\text{total}}` is the total number of all source characters observed in the data. The error space consists of insertions of any of the :math:`V` vocabulary characters.

Bayesian Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

When :math:`k > 0`, the parameter acts as the concentration parameter of a **symmetric Dirichlet prior distribution**. This represents a prior belief that every possible error is equally likely and has a "pseudo-count" of `k`.

Normalization
~~~~~~~~~~~~~

The costs are normalized by a ceiling :math:`Z` that depends on the vocabulary size. It is the a priori surprisal of any single error, assuming a uniform probability distribution over all possible outcomes.

.. math:: Z = -\log(\frac{1}{V+1}) = \log(V+1)

This normalization contextualizes the cost relative to the language's complexity. An error in a language with a large alphabet (high :math:`V`, e.g., Chinese) is less surprising than the same error in a language with a small alphabet (e.g., English).

Final Cost
~~~~~~~~~~

The final cost :math:`w(e)` is the base surprisal scaled by the normalization ceiling:

.. math:: w(e) = \frac{-\log(P(e))}{Z}

This cost is a relative measure. Costs can be greater than 1.0, which indicates the observed event was even less probable than the uniform a priori assumption.

Asymptotic Properties
~~~~~~~~~~~~~~~~~~~~~

As the amount of training data grows, the learned cost for an operation with a stable frequency ("share") converges to a fixed value - independent of :math:`k`:

.. math:: w(e) \approx \frac{-\log(\text{share})}{\log(V+1)}
