from reinsurance_classifier.aggregation import CriterionValue
from reinsurance_classifier.concordance import Vote, _outcome, _quorum_value


def test_quorum_requires_prompt_and_model_diversity() -> None:
    passing = (CriterionValue.PASS,) * 5
    diverse = [
        Vote(prompt=prompt, model=model, values=passing, valid=True)
        for prompt, model in (
            ("p1", "m1"),
            ("p1", "m2"),
            ("p2", "m2"),
            ("p2", "m3"),
        )
    ]
    assert _quorum_value(diverse, 0, quorum=4)[0] == CriterionValue.PASS
    one_prompt = [Vote("p1", f"m{i}", passing, True) for i in range(4)]
    assert _quorum_value(one_prompt, 0, quorum=4)[0] == CriterionValue.UNCLEAR


def test_outcome_rejects_on_a_resolved_failure() -> None:
    assert _outcome([CriterionValue.FAIL] + [CriterionValue.UNCLEAR] * 4) == "rejected"
    assert _outcome([CriterionValue.PASS] * 5) == "qualified"
