===================
Cost Learning Model
===================

The ``Learner`` class calculates edit costs using a probabilistic model. The cost of an edit operation is defined by its "surprisal": a measure of how unlikely the event is based on the training data.

The cost of an event :math:`e` is based on its negative log-likelihood. This value, also known as surprisal, quantifies the information contained in observing that event.

.. math:: \text{base_cost}(e) = -\log(P(e))

A common, high-probability error will have low surprisal and thus a low base cost. A rare, low-probability error will have high surprisal and a high base cost.

-------------------
Probabilistic Model
-------------------

The model estimates the probability of edit operations and transforms these probabilities into normalized, comparable costs. The smoothing parameter :math:`k` (set via ``with_smoothing()``) allows for a continuous transition between a Maximum Likelihood Estimation and a smoothed Bayesian model.

Probability for Substitutions and Deletions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an error :math:`e` (a substitution or deletion) involving a source character :math:`s`, the smoothed conditional probability is:

.. math:: P(e|s) = \frac{c_e + k}{C_s + k \cdot V_s}

- :math:`c_e`: The observed count of the specific error :math:`e`.
- :math:`C_s`: The total number of times the source character :math:`s` appeared.
- :math:`k`: The smoothing parameter.
- :math:`V_s`: The number of possible outcomes for source character :math:`s`. This is the vocabulary size plus one (for deletion).

Probability for Insertions
~~~~~~~~~~~~~~~~~~~~~~~~~~

For an insertion of a target character :math:`t`, the probability is conditioned on the entire corpus:

.. math:: P(ins(t)) = \frac{c_{ins(t)} + k}{C_{\text{total}} + k \cdot V_{ins}}

- :math:`c_{ins(t)}`: The observed count of inserting character :math:`t`.
- :math:`C_{\text{total}}`: The total number of all source characters observed.
- :math:`k`: The smoothing parameter.
- :math:`V_{ins}`: The number of possible characters that can be inserted (the vocabulary size).

Bayesian Perspective on Smoothing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When :math:`k > 0`, the parameter acts as the concentration parameter of a **symmetric Dirichlet prior distribution**. This represents a prior belief that every possible error has a "pseudo-count" of `k`. This regularizes the model by ensuring no probability is ever zero, which is especially useful for sparse data. The probability formula can be seen as:

.. math:: P(e|s) = \frac{\text{Observed Counts } + \text{ Prior Pseudo-Counts}}{\text{Total Observations } + \text{ Total Prior Pseudo-Counts}}

Normalization
~~~~~~~~~~~~~

To make surprisal values comparable and meaningful, they are normalized. The model uses a normalization ceiling, :math:`Z`, that is constant and depends only on the complexity (vocabulary size) of the language, not the amount of data.

This ceiling is the a priori surprisal of any single error in a world of maximum uncertainty (i.e., assuming a uniform probability distribution over all possible errors).

.. math:: Z = \log(V_s)

This normalization contextualizes the cost of an error. A specific error in a language with a large alphabet (e.g., Chinese) is inherently less surprising than the same error in a language with a small alphabet (e.g., English), and the normalization reflects this.

Final Cost
~~~~~~~~~~

The final cost is the base surprisal of the event, scaled by the normalization ceiling. This formula applies for all :math:`k \ge 0`.

.. math:: w(e) = \frac{-\log(P(e))}{Z}

This cost is a relative measure of surprisal. Costs can be greater than 1.0, which indicates that the observed event was even less probable than the uniform a priori assumption.

Asymptotic Properties
~~~~~~~~~~~~~~~~~~~~~

As the amount of training data grows infinitely large, the learned costs for an operation :math:`e` with a stable frequency ("share") converge to a fixed value that is independent of the dataset size. The asymptotic cost is:

.. math:: w(e) \approx \frac{-\log(\text{share})}{\log(V_s)}

This formula reveals two key properties of the model:

- **Dependence on Error Share:** The numerator, :math:`-\log(\text{share})`, is the empirical surprisal of the error.
- **Dependence on Vocabulary Size:** The denominator, :math:`\log(V_s)`, is the a priori surprisal based on the language's complexity. It acts as a constant baseline for surprise. The final cost is therefore the *empirical* surprisal measured relative to the *theoretical* complexity of the character set.
