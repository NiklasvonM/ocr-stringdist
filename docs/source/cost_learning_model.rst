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
- :math:`C(x)`: The total count of a specific context character :math:`x`. For example, :math:`C(s)` is the total number of times the source character :math:`s` appeared in the OCR outputs.
- :math:`V`: The total number of unique characters in the vocabulary.

Probability of an Edit Operation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model treats all edit operations within the same probabilistic framework. An insertion is modeled as a substitution from a ground-truth character to an "empty" character, and a deletion is a substitution from an OCR character to an empty character.

This means that for any given character (either from the source or the target), there are :math:`V+1` possible outcomes: a transformation into any of the :math:`V` vocabulary characters or a transformation into an empty character.

The smoothed conditional probability for any edit event :math:`e` given a context character :math:`x` (where :math:`x` is a source character for substitutions/deletions or a target character for insertions) is:

.. math:: P(e|x) = \frac{c(e) + k}{C(x) + k \cdot (V+1)}


Bayesian Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

When :math:`k > 0`, the parameter acts as the concentration parameter of a **symmetric Dirichlet prior distribution**. This represents a prior belief that every possible error is equally likely and has a "pseudo-count" of `k`.

Normalization
~~~~~~~~~~~~~

The costs are normalized by a ceiling :math:`Z` that depends on the size of the unified outcome space. It is the a priori surprisal of any single event, assuming a uniform probability distribution over all :math:`V+1` possible outcomes.

.. math:: Z = -\log(\frac{1}{V+1}) = \log(V+1)

This normalization contextualizes the cost relative to the complexity of the character set.

Final Cost
~~~~~~~~~~

The final cost :math:`w(e)` is the base surprisal scaled by the normalization ceiling:

.. math:: w(e) = \frac{-\log(P(e|x))}{Z}

This cost is a relative measure. Costs can be greater than 1.0, which indicates the observed event was less probable than the uniform a priori assumption.

Asymptotic Properties
~~~~~~~~~~~~~~~~~~~~~

As the amount of training data grows, the learned cost for an operation with a stable frequency ("share") converges to a fixed value - independent of :math:`k`:

.. math:: w(e) \approx \frac{-\log(\text{share})}{\log(V+1)}
