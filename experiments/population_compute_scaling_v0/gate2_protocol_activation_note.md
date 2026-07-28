# Gate-2 activation note

Gate 2 is activated from the positive Gate-1 v1 target-GPU result.

The frozen protocol is:

- `gate2_persistent_state_capacity_protocol_v0.md`

The critical design change relative to Gate 0 is that **source coverage and total learned update count are invariant across runtime population widths**. Population width changes only the number of persistent runtime neural state slots available to the same shared learned machinery.

No Gate-2 development or confirmation result existed when the protocol was frozen.
