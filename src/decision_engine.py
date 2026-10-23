"""Turn a fraud probability into a business decision."""

APPROVE = "approve"
REVIEW = "review"


def decide(probability, threshold):
    """Single threshold chosen on validation with the business cost function.
    Scores at or above it go to manual review; everything else is approved."""
    return REVIEW if probability >= threshold else APPROVE
