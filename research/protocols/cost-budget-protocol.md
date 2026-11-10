# GPU residence-time budget comparison

Date: 2026-11-06. This is a new comparison separate from the original 21-query matrix. Before this protocol was implemented, the first seed’s fixed, random, greedy, and BO development proxies and the full fixed-method confirmation had been seen; the main LLM result had not. This is not a fully prospective preregistration.

## Method sources and parts that remain unchanged

The experiment retains the LoRA training direction, the Task Arithmetic global-grid reference, the four-depth-group adaptation, and the declared simplified Gardner-style constrained BO baseline. The time budget changes scheduling and candidate availability, not the weighting algorithm. The method-to-tool contract, 64 development questions, 32 protection questions, 21 coefficient levels, bootstrap, and final held-out set retain their existing definitions. The old V2 source code and results are not overwritten.

Comparison of fixed global grid, random, greedy, BO, LLM; each method uses the same model and the same seed-specific direction. The fixed grid still only has 21 global points, and the budget cannot be wasted on duplicate points to create an appearance of equal cost. Other strategies have a maximum of 256 queries; if the query upper limit is hit first, the run is marked censored and the matrix is ​​stopped, which cannot be used as the completed time budget result.

## Clock and common budget

The new `Lab.gpu_lock_wait_seconds` only measures the waiting time for applying for the GPU0 file lock of this project and does not change the numerical operation. The new budget run starts at the main entry point; subtracting this wait yields active time. It includes input/source checking, target model loading, fingerprint and direction preparation, controller loading, all strategy decisions and target inference; interpreter/module import startup is not included, and previously shared LoRA training is reported separately. Therefore its accurate name is **GPU resident wall clock budget with loading**, not the full project economic cost, energy consumption, or cross-machine equivalent cost.

Each model/seed first reruns a complete 21-point fixed grid calibration; the total active time B of this run is used as a common budget for other strategies of the same model/seed. The fixed calibration reuses itself as a fixed reference without falsifying a second independent fixed run. B is not determined by the test accuracy, and the old setup time including queuing is not regarded as a pure loading cost.

Only candidates with completion time ≤ B are allowed to enter the optimal selection. If loading/decision-making runs out of B and no candidate is completed in time, the base will be returned unmodified and clearly marked "no observation completed within budget" without falsifying base measurements. Currently, a soft cutoff is used: the proposal/evaluation being executed may cross B, and its actual time consumption, tokens, and results are still recorded, but late-arriving candidates cannot be selected. The complete active time and overrun are reported separately.

This is a comparison of the "incumbent available at the common budget moment", **not strictly the same as the actual time spent**. If the paper requires the latter, interruptible execution/external monitoring and resource release time verification must also be added. You cannot remove the cost of late calls and wrap soft deadlines into strict budget control. Peak memory is also reported to show resource differences for resident controllers.

## Execution and Audit

`optimize_public_budget.py` saves candidate active time, deadline availability, strategy decision time, LLM input and output tokens and unexecuted proposals; `run_budget_matrix.py` freezes the source code and contract, runs a smoke test, then runs four models × three training seeds × five strategies. The smoke test uses 4 development and 4 protection questions, with a budget of at least 30 seconds. It only verifies execution and clock, and does not select formal hyperparameters based on this.

`audit_search.py` continues to support the original V2, and independently recalculates the candidate validity, protection statistics, deadline eligibility, optimal selection and base/delta binding of the new track. CPU tests check that better candidates arriving late are excluded, tampered deadline eligibility is rejected, and the base model is retained when no observation completes within budget. The audit only checks the consistency of the records, not independent retesting of GPU time.

After a complete search, the actual selected model still needs to be confirmed on the fixed evaluation backend, batch and data, and the mathematically equivalent parameter state in the main confirmation will be reused first; its timing must not be reused for new search timing. Development proxy improvements are reported separately from actual task improvements. This track has not yet been completed and no cost advantages have been concluded.

## Full held-out set confirmation

`run_budget_confirmation.py` has been submitted and is waiting for the original main confirmation and the budget search matrix to be completed. Use the frozen benchmark runner and all its helpers used in the original main confirmation without changing the evaluation backend. Includes complete GSM8K (batch32), ARC-C and HellaSwag (batch8), and reuses the base model and standard LoRA reference. The same cross-model/seed statistics script is automatically run upon completion; the search completion mark itself does not count as quality confirmation completion.

`confirmation_state.py` identifies parameter state from base model/tokenizer fingerprint, delta manifest and compilation coefficient. The integer 1 and the floating point 1.0 will be converted to the same float by the actual compiler, so they are expressed uniformly; non-zero coefficients are not approximately rounded. An all-zero plan skips delta hooks in this executor, allowing reuse of the native base. Different weights/non-zero coefficients cannot be combined based on similarity indicators or tensor distances.

Existing results must pass complete per-question scoring, source code, execution settings and status binding checks; if different directories represent the same status but give different per-question results, reuse will be stopped and investigation required. New states are fully inferred independently; existing but incomplete outputs are retained and stopped and are not restarted due to observation timeouts. All reused paths and markers are written to the index, no new timings or new independent repeats are created.

The current four CPU checks have passed. The actual three bindings of the original Qwen0.6 model, standard LoRA and 0.8 selected states are different from each other. The selected plan is consistent with the completed official result binding. Full budget confirmation has not yet begun inference and these checks cannot be called end-to-end results.
