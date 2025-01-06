import numpy as np
import pytest

from theory.method.detailed_distribution import DetailedDistribution
from theory.method.detailed_eval import DetailedEvaluator
from theory.method.distribution import (
    DetailedDistributionMaker,
    ValuedDistribution,
    ValuedDistributionMaker,
)
from theory.method.evaluate import Evaluator
from theory.method.preeval import (
    calc_attack_string_mapping,
    calc_eas_mapping,
    calc_postfix_mapping,
)

EPSILON = 1e-10


arguments_cons = [
    (np.float64(0.451), 1, 4),
    (np.float64(0.201), 2, 3),
    (np.float64(0.321), 2, 2),
]


@pytest.mark.parametrize("alpha,size_prefix,size_postfix", arguments_cons)
def test_consistency_null_iteration(
    alpha: np.float64, size_prefix: int, size_postfix: int
):
    known, attack_string_mapping = calc_attack_string_mapping(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
    )
    eas_mapping = calc_eas_mapping(
        attack_string_mapping=attack_string_mapping,
        postfix_mapping=calc_postfix_mapping(attack_string_mapping),
    )

    detailed_evaluator = DetailedEvaluator[DetailedDistribution](
        maker=DetailedDistributionMaker(
            alpha=alpha,
            size_prefix=size_prefix,
            size_postfix=size_postfix,
            eas_mapping=eas_mapping,
        ),
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        known=known,
        eas_mapping=eas_mapping,
    )

    value_function = {str(eas): np.float64(0.0) for eas in detailed_evaluator.eass}

    detailed_dists = detailed_evaluator.eval_all_eas(value_function=value_function)
    detailed_exp_vals = {
        eas: dist.expected_value() for eas, dist in detailed_dists.items()
    }

    maker = ValuedDistributionMaker(
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        eas_mapping=eas_mapping,
    )
    evaluator = Evaluator[ValuedDistribution](
        alpha=alpha,
        size_prefix=size_prefix,
        size_postfix=size_postfix,
        known=known,
        eas_mapping=eas_mapping,
        optimize=False,
    )

    dists = evaluator.eval_all_eas(value_function=value_function, maker=maker)
    exp_vals = {eas: dist.expected_value() for eas, dist in dists.items()}

    assert set(detailed_exp_vals) == set(exp_vals)
    for eas, val in exp_vals.items():
        assert np.isclose(
            val, detailed_exp_vals[eas], rtol=EPSILON
        ), f"{eas=} {val=} {detailed_dists[eas]=}"

    for eas, dist in dists.items():
        for elem, det_elem in zip(dist.distribution, detailed_dists[eas].distribution):
            assert np.isclose(
                elem.value, det_elem.value, rtol=EPSILON
            ), f"{eas=} {elem.value=} {det_elem.value=} {len(dist.distribution)=} {len(detailed_dists[eas].distribution)=} {elem.probability=} {det_elem.probability=}"
            assert np.isclose(
                elem.probability, det_elem.probability, rtol=EPSILON
            ), f"{eas=} {elem.probability=} {det_elem.probability=}"
