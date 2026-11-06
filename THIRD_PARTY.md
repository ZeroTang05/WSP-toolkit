# Third-party method and source attribution

Original project code is released under Apache-2.0. This does not relicense
upstream code, model checkpoints, benchmark tasks or research logs.

- SparseGPT, Elias Frantar and Dan Alistarh, ICML 2023:
  https://arxiv.org/abs/2301.00774v3. The reconstruction module is a declared
  adaptation of its algorithm and Apache-2.0 reference implementation. Source
  blob and changes are in `research/references/sparsegpt/provenance.json` and
  the module docstring. Its license is bundled under `weight_programming/licenses`.
- MIT WSP experiment 8: https://github.com/RikiyaT/lm_understand-logs.
  Boxed-answer extraction is adapted from the MIT-licensed `mathqa.py`, blob
  808a04776906b1cf4630d397cef54f03d8795477. Copyright and license are retained
  in the grading module and bundled license. Research logs/data in that
  repository have their own CC BY 4.0 terms; they are reference material, not
  independent-discovery inputs or bundled training data.
- OfficeBench source and benchmark: https://github.com/zlwang-cs/OfficeBench,
  commit b978b808667c32b52ce19a67ce1def1de9ae02b7, Apache-2.0. Acquisition
  preserves its license and source. The container execution port declares
  environment, transport and network changes separately from official scores.
- Existing `vendor/` sources retain their own LICENSE and provenance files.
  Method inspiration alone is not a claim of code reproduction or validation.

Model and dataset licenses remain the responsibility of their respective
distributions. Wheels contain no model checkpoint, reference weight-edit solution
or professional benchmark answers. The research workspace contains separately
acquired public reference artifacts; do not mount those into a blind programmer.
