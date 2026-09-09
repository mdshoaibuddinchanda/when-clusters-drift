# Hypothesis Y1 convergence policy

Status: fixed before any corrected-replication outcome was inspected. This policy does not revise the frozen Phase 7 result.

The original `usable` gate accepted 635 rows for which at least one of the source or candidate FCM fits had not converged. That is a methodological validity defect: successful serialization and finite output are not evidence that an iterative optimum was reached.

Diagnostic reruns used the original data, initialization, fuzzifier rule, and tolerance, changing only the iteration horizon for characterization. Among 161 originally failed fits sampled under the documented coverage rule, 110 converged by iteration 300 and 153 by iteration 600. None showed a material objective increase, numerical failure, empty cluster, or degenerate solution. The dominant failure mode is therefore slow convergence / truncation at 150, not objective oscillation or a demonstrated FCM implementation error. Eight sampled failures remained unresolved by 600.

For corrected Y1 work, `max_iter=600` and `tol=1e-5` are fixed. A row is primary-eligible only when both its source and candidate fits have `SUCCESS_CONVERGED`. Near-converged runs (finite final center shift no more than ten times tolerance, monotone objective, otherwise valid) are retained only for a named sensitivity analysis. All other non-converged, degenerate, numerical, or empty-cluster results are excluded from primary estimation and counted explicitly. A dataset retaining less than 80% of its planned scenarios is marked insufficient; it is never silently deleted.

The choice of 600 is based solely on the convergence diagnostic and not on whether P4 improves. The tolerance, initialization, fuzzifier policy, dataset panel, and scientific outcome threshold are unchanged.
