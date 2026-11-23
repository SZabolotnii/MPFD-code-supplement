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
#print axioms MCP.Reframe.bordered_solve
#print axioms MCP.Reframe.bordered_solve_witness
#print axioms MCP.Reframe.same_parity_product
#print axioms MCP.Reframe.same_parity_product_witness
#print axioms MCP.Reframe.opposite_parity_product
#print axioms MCP.Reframe.same_parity_sum_sign
#print axioms MCP.Reframe.same_parity_sum_sign_witness
#print axioms MCP.Reframe.rho_props
#print axioms MCP.Reframe.rho_props_witness
#print axioms MCP.Reframe.crossover_balance
#print axioms MCP.Reframe.crossover_balance_witness
