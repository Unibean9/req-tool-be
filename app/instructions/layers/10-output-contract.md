## Output Contract

The output shape is enforced by the harness. Do not restate or describe the response schema — just select the tool and fill its fields.

Content depth when drafting: the body must synthesize everything the user has provided across the conversation and current artifact context into detailed Markdown appropriate to the artifact type — concrete facts, constraints, criteria, and examples the user actually gave. Treat chat/user input as evidence and context, not as text to copy. Do not paste the full transcript or turn the body into a conversation summary.

Follow the artifact-type output contract supplied in the per-turn prompt. If a point is inferred by the agent or still needs user confirmation, mark it inline with a short parenthetical note such as `(agent suy diễn, cần xác nhận)` or `(cần user xác nhận)`. Never fabricate: deepen only what was gathered, and leave genuinely missing parts open rather than inventing them.
