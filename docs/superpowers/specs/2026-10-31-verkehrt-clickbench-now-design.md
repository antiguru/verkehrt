# ClickBench queries on today's primitives

This spec adds the ClickBench queries that the tier 2 primitives can express to the acceptance suite. It is tracked as issue #20 and comes before plan 5, so the gaps it finds can shape plan 5's scope. It extends `2026-10-31-verkehrt-tier2-design.md`, whose acceptance method it reuses unchanged.

## Why

The acceptance suite covers ClickBench q01, q02, q06, q07, and q42, and TPC-H q01 and q06. Corgi's notes in `algorithms/clickbench/README.md` suggest that nine more ClickBench queries need nothing beyond the tier 2 primitives: q00, q03, q19, q27, q29, q30, q31, q40, and q41. That estimate comes from the query shapes, not from written programs. Writing them either widens the suite or turns up concrete gaps, the way q42 turned up the unshifted bucket count in plan 4.

## Goals

* Each of the nine queries gets a program in `tests/acceptance/clickbench/` whose output matches DuckDB's answer in `expected/` byte for byte, through the existing harness. The acceptance test and the acceptance bench discover the programs automatically.
* A query that cannot match exactly with today's primitives gets no program. Instead, the gap is recorded in the measurement document and filed as an issue, with the primitive or semantics it needs.
* The new queries are timed against corgi and DuckDB on one thread, with the method of `docs/measurements/2026-10-31-tier2.md`, on the x86 benchmark host. The results go into a new measurement document with raw output beside it.
* A query that is correct but slower than DuckDB or corgi is recorded with the kernel that dominates its time (from `--stats`), and grouped under the plan that would address it (plan 5 for sort-bound grouping, issue #16 for mask chains).

## Non-goals

* No new primitives or kernels. A query that needs one is a gap, not a reason to extend the language here.
* No changes to the harness or to `prepare.py`'s output format, except that the harness may gain an output kind that `prepare.py` already writes and the harness does not yet read. That change gets a unit test like the existing kinds.
* No performance work. Slow but correct is an acceptable outcome, because the measurement's purpose is to feed plan 5.

## Query notes

These notes are starting points from the SQL and corgi's versions. The builder confirms each against the data.

* q00 is a row count, written as `length` of one column.
* q19 filters `UserID` against a constant. The expected answer is empty on this data, and the program must still compute it.
* q29 is ninety sums of `ResolutionWidth + k`. It is written as the SQL is, with no algebraic shortcut, so it measures ninety passes, as corgi's version does.
* q41's `OFFSET 10000` makes the expected answer empty, and the program must still compute the grouping.
* q03 averages `UserID`, whose sum exceeds `i64`. Corgi splits each id at 2^32 and sums the halves separately. The program must reproduce DuckDB's printed value exactly, including its formatting.
* q27 needs URL lengths, which come from the string column's `ends` offsets. It groups by `CounterID`, applies the scaled `HAVING`, and orders by the average length descending with `CounterID` as the tiebreak.
* q30, q31, and q40 group by keys without a small range: `ClientIP`, `WatchID`, and `URLHash`. They group by sorting with `argsort`, which is stable, so a multi-column key sorts by the minor column first and the major column second, and run starts come from `prev` and a comparison. Their time is expected to be dominated by `argsort` (issue #9).

## Left to the builder

* How each query is decomposed into primitives, within the language as it stands.
* Whether a gap is real or a missing idiom. The builder checks the existing programs and `tests/programs/` before declaring a gap.
