# Gate-9D learned shared router hard-negative v1

Development-only correction of the supervised routing experiment.

The v0 router learned the zero-worker gate but collapsed the contribution gate
to nearly all basis workers, because aggregate accuracy concealed the rare
selected-basis predicate. V1 keeps the same shared router and execution
semantics, but trains on an explicitly balanced local routing curriculum:

- zero-input workers;
- selected basis workers;
- unselected basis workers;
- non-basis distractors.

Qualification and result reporting must include confusion counts, selected-basis
recall, unselected-basis specificity, distractor specificity, exact gate
accuracy, expected versus observed messages per episode, full execution,
permutation execution, and shuffled-support control.

This remains supervised routing. It does not establish automatic coordinate
discovery or end-to-end learning from answer loss.
