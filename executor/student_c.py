# Owner: Student C
"""Working C starting point. Extend the shared closed-loop simulation skills here."""
from executor.closed_loop import ClosedLoopExecutor


class StudentCExecutor(ClosedLoopExecutor):
    """Eight bounded skills, with public sensor checks and one local grasp retry.

    Tested initially with weld attachment. This flag enables the interface;
    it does not certify contact-only grasping or the full course evaluation.
    """

    IMPLEMENTED = True
    LABEL = "C: Student C closed-loop baseline (simulation / weld)"
