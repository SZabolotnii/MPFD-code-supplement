import Mathlib

/-!
# Parity block structure of the deflection (Letter, L2)

Metric `F = fromBlocks A 0 0 D` (even/odd blocks decouple for a symmetric pair),
target `Y = Sum.elim y 0` (odd block of the target vanishes). Then
`Y ⬝ᵥ F⁻¹ *ᵥ Y = y ⬝ᵥ A⁻¹ *ᵥ y`: the odd columns contribute nothing.
-/

open Matrix

namespace MCP

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]

/-- Inverse of a block-diagonal matrix (both blocks invertible). -/
lemma inv_fromBlocks_diag (A : Matrix m m ℝ) (D : Matrix n n ℝ)
    (hA : IsUnit A.det) (hD : IsUnit D.det) :
    (fromBlocks A 0 0 D)⁻¹ = fromBlocks A⁻¹ 0 0 D⁻¹ := by
  have hF : IsUnit (fromBlocks A 0 0 D).det := by
    rw [det_fromBlocks_zero₂₁]; exact hA.mul hD
  apply inv_eq_right_inv
  rw [fromBlocks_multiply]
  simp [mul_nonsing_inv _ hA, mul_nonsing_inv _ hD, fromBlocks_one]

/-- J of a block-diagonal metric with target supported on the first block. -/
theorem deflection_block_diag (A : Matrix m m ℝ) (D : Matrix n n ℝ)
    (hA : IsUnit A.det) (hD : IsUnit D.det) (y : m → ℝ) :
    let F := fromBlocks A 0 0 D
    let Y : m ⊕ n → ℝ := Sum.elim y 0
    Y ⬝ᵥ (F⁻¹ *ᵥ Y) = y ⬝ᵥ (A⁻¹ *ᵥ y) := by
  intro F Y
  have h := inv_fromBlocks_diag A D hA hD
  simp only [F, Y, h, fromBlocks_mulVec, Sum.elim_comp_inl, Sum.elim_comp_inr,
    mulVec_zero, add_zero, Matrix.zero_mulVec]
  simp [dotProduct, Fintype.sum_sum_type]

/-- Odd columns contribute nothing when the odd part of the target is zero:
adding them to an even dictionary leaves J unchanged. -/
theorem odd_block_contributes_nothing (A : Matrix m m ℝ) (D : Matrix n n ℝ)
    (hA : IsUnit A.det) (hD : IsUnit D.det) (y : m → ℝ) :
    (Sum.elim y (0 : n → ℝ)) ⬝ᵥ ((fromBlocks A 0 0 D)⁻¹ *ᵥ Sum.elim y 0) = y ⬝ᵥ (A⁻¹ *ᵥ y) :=
  deflection_block_diag A D hA hD y

end MCP
