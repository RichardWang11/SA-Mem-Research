# Repaired-memory QA / LLM-J audit, 2026-10-25

Read [REPORT.md](REPORT.md) for the paired results and failure cases.

- `qa_and_request_trace.tar.gz`: all 321 new answers, retrieval evidence and
  traces, two runtime smoke tests, 646 completed logical API traces (323 answer
  and 323 query embedding calls), per-Conversation console logs, observed code
  snapshot and offline lexical scores. Queued Conversations computed ahead
  were reused, so duplicate physical working directories are not archived.
- `judge_after.tar.gz`: 321 new per-question judge results, all raw request /
  response attempts, inputs, requested settings and original runner snapshot.
- `judge_before.tar.gz`: the 321 original baseline judge results and requests;
  these were reused, not rejudged. Its original summary has empty subset groups
  because old records lack a subset field; use the reaudited summary below.
- `baseline_judge_summary_reaudited.json` and
  `repaired_judge_summary_reaudited.json`: canonical CSV membership applied to
  original completed labels. Original answers/input hashes and equality of
  Judge prompt hash/model/endpoint/temperature/output settings were checked.
- `label_changes.json`: every QID's unchanged before/after label and subset.
- `evidence_samples.json`, `evidence_summary.json`: source-mapped Gold recall.
- `failure_cases.json`: all 75 wrong main-subset samples with evidence status.
- `scripts/`: observed local rerun/analysis tools. Their server paths and
  credential-file selectors describe this run; use the portable entrypoints in
  `../../memory_repair/` on another machine.
- `files_sha256.json`: archive and loose-file integrity checks.

321 QA and 321 Judge results are valid; all answer completions stopped normally.
The original coordinator encountered a post-processing error while subtracting
an empty historical subset summary. This was recovered without any new API
calls by reaggregating the saved original labels against canonical annotations.
The archived coordinator script includes that post-processing fix; the actual
inference source snapshot is in the QA archive and was not changed during QA.

Judge requested model: `gpt-4o-mini`; returned metadata was
`gpt-4o-mini-2024-07-18` for 312 responses and `gpt-4o-mini` for 9.
Prompt: [Mem0 Appendix A text](judge_prompt.txt). This is not the TSM paper's
GPT-4.1-mini judge or a full 1,986-question LoCoMo result.

Adversarial uses a labelled synthetic abstention reference and is reported
separately. Both ordinary-QA improvement and adversarial degradation are kept.
No Temporal-first method was implemented or measured in this comparison.
