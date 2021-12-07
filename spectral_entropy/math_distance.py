import numpy as np
import scipy.stats


def unweighted_entropy_distance(p, q):
    r"""
    Unweighted entropy distance:

    .. math::

        -\frac{2\times S_{PQ}-S_P-S_Q} {ln(4)}, S_I=\sum_{i} {I_i ln(I_i)}
    """
    merged = p + q
    entropy_increase = 2 * \
                       scipy.stats.entropy(merged) - scipy.stats.entropy(p) - \
                       scipy.stats.entropy(q)
    return entropy_increase


def entropy_distance(p, q):
    r"""
    Entropy distance:

    .. math::

          -\frac{2\times S_{PQ}^{'}-S_P^{'}-S_Q^{'}} {ln(4)}, S_I^{'}=\sum_{i} {I_i^{'} ln(I_i^{'})}, I^{'}=I^{w}, with\ w=0.25+S\times 0.5\ (S<1.5)
    """
    p = _weight_intensity_by_entropy(p)
    q = _weight_intensity_by_entropy(q)

    return unweighted_entropy_distance(p, q)


def _weight_intensity_by_entropy(x):
    WEIGHT_START = 0.25
    ENTROPY_CUTOFF = 3
    weight_slope = (1 - WEIGHT_START) / ENTROPY_CUTOFF

    if np.sum(x) > 0:
        entropy_x = scipy.stats.entropy(x)
        if entropy_x < ENTROPY_CUTOFF:
            weight = WEIGHT_START + weight_slope * entropy_x
            x = np.power(x, weight)
            x_sum = np.sum(x)
            x = x / x_sum
    return x


def _select_common_peaks(p, q):
    select = q > 0
    p = p[select]
    p_sum = np.sum(p)
    if p_sum > 0:
        p = p / p_sum
    q = q[select]
    q = q / np.sum(q)
    return p, q


def euclidean_distance(p, q):
    r"""
    Euclidean distance:

    .. math::

        (\sum|P_{i}-Q_{i}|^2)^{1/2}
    """
    return np.sqrt(np.sum(np.power(p - q, 2)))


def manhattan_distance(p, q):
    r"""
    Manhattan distance:

    .. math::

        \sum|P_{i}-Q_{i}|
    """
    return np.sum(np.abs(p - q))


def chebyshev_distance(p, q):
    r"""
    Chebyshev distance:

    .. math::

        \underset{i}{\max}{(|P_{i}\ -\ Q_{i}|)}
    """
    return np.max(np.abs(p - q))


def squared_euclidean_distance(p, q):
    r"""
    Squared Euclidean distance:

    .. math::

        \sum(P_{i}-Q_{i})^2
    """
    return np.sum(np.power(p - q, 2))


def fidelity_distance(p, q):
    r"""
    Fidelity distance:

    .. math::

        1-\sum\sqrt{P_{i}Q_{i}}
    """
    return 1 - np.sum(np.sqrt(p * q))


def matusita_distance(p, q):
    r"""
    Matusita distance:

    .. math::

        \sqrt{\sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2}
    """
    return np.sqrt(np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2)))


def squared_chord_distance(p, q):
    r"""
    Squared-chord distance:

    .. math::

        \sum(\sqrt{P_{i}}-\sqrt{Q_{i}})^2
    """
    return np.sum(np.power(np.sqrt(p) - np.sqrt(q), 2))


def bhattacharya_1_distance(p, q):
    r"""
    Bhattacharya 1 distance:

    .. math::

        (\arccos{(\sum\sqrt{P_{i}Q_{i}})})^2
    """
    s = np.sum(np.sqrt(p * q))
    if s > 1:
        s = 1
    return np.power(np.arccos(s), 2)


def bhattacharya_2_distance(p, q):
    r"""
    Bhattacharya 2 distance:

    .. math::

        -\ln{(\sum\sqrt{P_{i}Q_{i}})}
    """
    s = np.sum(np.sqrt(p * q))
    if s == 0:
        return np.inf
    else:
        return -np.log(s)


def harmonic_mean_distance(p, q):
    r"""
    Harmonic mean distance:

    .. math::

        1-2\sum(\frac{P_{i}Q_{i}}{P_{i}+Q_{i}})
    """
    return 1 - 2 * np.sum(p * q / (p + q))


def probabilistic_symmetric_chi_squared_distance(p, q):
    r"""
    Probabilistic symmetric χ2 distance:

    .. math::

        \frac{1}{2} \times \sum\frac{(P_{i}-Q_{i}\ )^2}{P_{i}+Q_{i}\ }
    """
    return 1 / 2 * np.sum(np.power(p - q, 2) / (p + q))


def ruzicka_distance(p, q):
    r"""
    Ruzicka distance:

    .. math::

        \frac{\sum{|P_{i}-Q_{i}|}}{\sum{\max(P_{i},Q_{i})}}
    """
    dist = np.sum(np.abs(p - q)) / np.sum(np.maximum(p, q))
    return dist


def roberts_distance(p, q):
    r"""
    Roberts distance:

    .. math::

        1-\sum\frac{(P_{i}+Q_{i})\frac{\min{(P_{i},Q_{i})}}{\max{(P_{i},Q_{i})}}}{\sum(P_{i}+Q_{i})}
    """
    return 1 - np.sum((p + q) / np.sum(p + q) * np.minimum(p, q) / np.maximum(p, q))


def intersection_distance(p, q):
    r"""
    Intersection distance:

    .. math::

        1-\frac{\sum\min{(P_{i},Q_{i})}}{\min(\sum{P_{i},\sum{Q_{i})}}}
    """
    return 1 - np.sum(np.minimum(p, q)) / min(np.sum(p), np.sum(q))


def motyka_distance(p, q):
    r"""
    Motyka distance:

    .. math::

        -\frac{\sum\min{(P_{i},Q_{i})}}{\sum(P_{i}+Q_{i})}
    """
    dist = np.sum(np.minimum(p, q)) / np.sum(p + q)
    return -dist


def canberra_distance(p, q):
    r"""
    Canberra distance:

    .. math::

        \sum\frac{|P_{i}-Q_{i}|}{|P_{i}|+|Q_{i}|}
    """
    return np.sum(np.abs(p - q) / (np.abs(p) + np.abs(q)))


def baroni_urbani_buser_distance(p, q):
    r"""
    Baroni-Urbani-Buser distance:

    .. math::

        1-\frac{\sum\min{(P_i,Q_i)}+\sqrt{\sum\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}{\sum{\max{(P_i,Q_i)}+\sqrt{\sum{\min{(P_i,Q_i)}\sum(\max{(P)}-\max{(P_i,Q_i)})}}}}
    """
    if np.max(p) < np.max(q):
        p, q = q, p
    d1 = np.sqrt(np.sum(np.minimum(p, q) * np.sum(max(p) - np.maximum(p, q))))
    return 1 - (np.sum(np.minimum(p, q)) + d1) / (np.sum(np.maximum(p, q)) + d1)


def penrose_size_distance(p, q):
    r"""
    Penrose size distance:

    .. math::

        \sqrt N\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return np.sqrt(n) * np.sum(np.abs(p - q))


def mean_character_distance(p, q):
    r"""
    Mean character distance:

    .. math::

        \frac{1}{N}\sum{|P_i-Q_i|}
    """
    n = np.sum(p > 0)
    return 1 / n * np.sum(np.abs(p - q))


def lorentzian_distance(p, q):
    r"""
    Lorentzian distance:

    .. math::

        \sum{\ln(1+|P_i-Q_i|)}
    """
    return np.sum(np.log(1 + np.abs(p - q)))


def penrose_shape_distance(p, q):
    r"""
    Penrose shape distance:

    .. math::

        \sqrt{\sum((P_i-\bar{P})-(Q_i-\bar{Q}))^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(np.sum(np.power((p - p_avg) - (q - q_avg), 2)))


def clark_distance(p, q):
    r"""
    Clark distance:

    .. math::

        (\frac{1}{N}\sum(\frac{P_i-Q_i}{|P_i|+|Q_i|})^2)^\frac{1}{2}
    """
    n = np.sum(p > 0)
    return np.sqrt(1 / n * np.sum(np.power((p - q) / (np.abs(p) + np.abs(q)), 2)))


def hellinger_distance(p, q):
    r"""
    Hellinger distance:

    .. math::

        \sqrt{2\sum(\sqrt{\frac{P_i}{\bar{P}}}-\sqrt{\frac{Q_i}{\bar{Q}}})^2}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return np.sqrt(2 * np.sum(np.power(np.sqrt(p / p_avg) - np.sqrt(q / q_avg), 2)))


def whittaker_index_of_association_distance(p, q):
    r"""
    Whittaker index of association distance:

    .. math::

        \frac{1}{2}\sum|\frac{P_i}{\bar{P}}-\frac{Q_i}{\bar{Q}}|
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    return 1 / 2 * np.sum(np.abs(p / p_avg - q / q_avg))


def symmetric_chi_squared_distance(p, q):
    r"""
    Symmetric χ2 distance:

    .. math::

        \sqrt{\sum{\frac{\bar{P}+\bar{Q}}{N(\bar{P}+\bar{Q})^2}\frac{(P_i\bar{Q}-Q_i\bar{P})^2}{P_i+Q_i}\ }}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)
    n = np.sum(p > 0)

    d1 = (p_avg + q_avg) / (n * np.power(p_avg + q_avg, 2))
    return np.sqrt(d1 * np.sum(np.power(p * q_avg - q * p_avg, 2) / (p + q)))


def pearson_correlation_distance(p, q):
    r"""
    Pearson/Spearman Correlation Coefficient:

    .. math::

        \frac{\sum[(Q_i-\bar{Q})(P_i-\bar{P})]}{\sqrt{\sum(Q_i-\bar{Q})^2\sum(P_i-\bar{P})^2}}
    """
    p_avg = np.mean(p)
    q_avg = np.mean(q)

    x = np.sum((q - q_avg) * (p - p_avg))
    y = np.sqrt(np.sum(np.power(q - q_avg, 2)) * np.sum(np.power(p - p_avg, 2)))

    if x == 0 and y == 0:
        return 0.
    else:
        return -x / y


def improved_similarity_distance(p, q):
    r"""
    Improved Similarity Index:

    .. math::

        \sqrt{\frac{1}{N}\sum\{\frac{P_i-Q_i}{P_i+Q_i}\}^2}
    """
    n = np.sum(p > 0)
    return np.sqrt(1 / n * np.sum(np.power((p - q) / (p + q), 2)))


def absolute_value_distance(p, q):
    r"""
    Absolute Value Distance:

    .. math::

        \frac { \sum(|Q_i-P_i|)}{\sum P_i}

    """
    dist = np.sum(np.abs(q - p)) / np.sum(p)
    return dist


def dot_product_distance(p, q):
    r"""
    Dot product distance:

    .. math::

        1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
    """
    score = np.power(np.sum(q * p), 2) / \
            (np.sum(np.power(q, 2)) * np.sum(np.power(p, 2)))
    return 1 - np.sqrt(score)


def cosine_distance(p, q):
    r"""
    Cosine distance, it gives the same result as the dot product.

    .. math::

        1 - \sqrt{\frac{(\sum{Q_iP_i})^2}{\sum{Q_i^2\sum P_i^2}}}
    """
    return dot_product_distance(p, q)


def dot_product_reverse_distance(p, q):
    r"""
    Reverse dot product distance, only consider peaks existed in spectrum Q.

    .. math::

        1 - \sqrt{\frac{(\sum{{} {P_i^{'}}})^2}{{\sum{(Q_i^{'})^2}{\sum (P_i^{'})^2}}}}, with:

        P^{'}_{i}=\frac{P^{''}_{i}}{\sum_{i}{P^{''}_{i}}},

        P^{''}_{i}=\begin{cases}
        0 & \text{ if } Q_{i}=0 \\
        P_{i} & \text{ if } Q_{i}\neq0
        \end{cases}

    """

    p, q = _select_common_peaks(p, q)
    if np.sum(p) == 0:
        score = 0
    else:
        score = np.power(np.sum(q * p), 2) / \
                (np.sum(np.power(q, 2)) * np.sum(np.power(p, 2)))
    return 1 - np.sqrt(score)


def spectral_contrast_angle_distance(p, q):
    r"""
    Spectral Contrast Angle distance.
    Please note that the value calculated here is :math:`\cos\theta`.
    If you want to get the :math:`\theta`, you can calculate with: :math:`\arccos(1-distance)`

    .. math::

        1 - \frac{\sum{Q_iP_i}}{\sqrt{\sum Q_i^2\sum P_i^2}}
    """
    return 1 - np.sum(q * p) / \
           np.sqrt(np.sum(np.power(q, 2)) * np.sum(np.power(p, 2)))


def wave_hedges_distance(p, q):
    r"""
    Wave Hedges distance:

    .. math::

        \sum\frac{|P_i-Q_i|}{\max{(P_i,Q_i)}}
    """
    return np.sum(np.abs(p - q) / np.maximum(p, q))


def jaccard_distance(p, q):
    r"""
    Jaccard distance:

    .. math::

        \frac{\sum(P_i-Q_i)^2}{\sum P_i^2+\sum{Q_i^2-\sum{P_iQ_i}}}
    """
    return np.sum(np.power(p - q, 2)) / \
           (np.sum(np.power(p, 2)) + np.sum(np.power(q, 2)) - np.sum(p * q))


def dice_distance(p, q):
    r"""
    Dice distance:

    .. math::

        \frac{\sum(P_i-Q_i)^2}{\sum P_i^2+\sum Q_i^2}
    """
    return np.sum(np.power(p - q, 2)) / \
           (np.sum(np.power(p, 2)) + np.sum(np.power(q, 2)))


def inner_product_distance(p, q):
    r"""
    Inner Product distance:

    .. math::

        1-\sum{P_iQ_i}
    """
    return 1 - np.sum(p * q)


def divergence_distance(p, q):
    r"""
    Divergence distance:

    .. math::

        2\sum\frac{(P_i-Q_i)^2}{(P_i+Q_i)^2}
    """
    return 2 * np.sum((np.power(p - q, 2)) / np.power(p + q, 2))


def avg_l_distance(p, q):
    r"""
    Avg (L1, L∞) distance:

    .. math::

        \frac{1}{2}(\sum|P_i-Q_i|+\underset{i}{\max}{|P_i-Q_i|})
    """
    return (np.sum(np.abs(p - q)) + max(np.abs(p - q)))


def vicis_symmetric_chi_squared_3_distance(p, q):
    r"""
    Vicis-Symmetric χ2 3 distance:

    .. math::

        \sum\frac{(P_i-Q_i)^2}{\max{(P_i,Q_i)}}
    """
    return np.sum(np.power(p - q, 2) / np.maximum(p, q))
