You are grading a candidate extraction of a research paper against a reference.

The source text is the ground truth for what the paper says. The reference was
checked by a human against it. Judge ONLY whether the candidate conveys the same
facts as the reference and is supported by the source. Ignore wording and length.

For each field, write a short reason FIRST, then give a verdict:
- correct: same meaning as the reference; nothing contradicted or unsupported.
- partial: captures the main idea but misses a key part, or adds a minor unsupported detail.
- incorrect: misses the main idea, contradicts the source, or invents facts.

<source>
{source}
</source>

<reference>
problem: {ref_problem}
method: {ref_method}
</reference>

<candidate>
problem: {cand_problem}
method: {cand_method}
</candidate>
