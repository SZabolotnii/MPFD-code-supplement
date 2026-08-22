import MCP

/-!
# Axiom audit: every theorem the Short Communication relies on must depend only on Lean's
three standard axioms (`propext`, `Classical.choice`, `Quot.sound`) -- never on `sorryAx`.
Run with:  lake env lean Audit.lean -/
#print axioms MCP.inv_fromBlocks_diag
#print axioms MCP.deflection_block_diag
#print axioms MCP.odd_block_contributes_nothing
#print axioms MCP.deflection_border_general
#print axioms MCP.deflection_border
#print axioms MCP.deflection_border_mono
#print axioms MCP.deflection_border_strict
