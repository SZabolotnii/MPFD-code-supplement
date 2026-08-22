import Mathlib

/-!
# Adding one column: Schur form of the deflection gain (Letter, L4)

`F = border A b d = fromBlocks A (col b) (row b) d`, `Y = Sum.elim y η`.
With `S = d - bᵀA⁻¹b ≠ 0`,
`YᵀF⁻¹Y = yᵀA⁻¹y + (η - bᵀA⁻¹y)(η - yᵀA⁻¹b) / S`,
which for a symmetric metric `A` is `yᵀA⁻¹y + (η - bᵀA⁻¹y)² / S`.
Strict gain when the residual `η - bᵀA⁻¹y ≠ 0` (this direction is the one proved).
-/

open Matrix

namespace MCP

variable {m : Type*} [Fintype m] [DecidableEq m]

/-- Bordered matrix: `A` with one extra row/column `b` and corner `d`. -/
noncomputable def border (A : Matrix m m ℝ) (b : m → ℝ) (d : ℝ) :
    Matrix (m ⊕ Unit) (m ⊕ Unit) ℝ :=
  fromBlocks A (replicateCol Unit b) (replicateRow Unit b) (of fun _ _ => d)

omit [DecidableEq m] in
/-- Action of the bordered matrix on a bordered vector. -/
lemma border_mulVec (A : Matrix m m ℝ) (b : m → ℝ) (d : ℝ) (x : m → ℝ) (c : ℝ) :
    border A b d *ᵥ Sum.elim x (fun _ => c)
      = Sum.elim (A *ᵥ x + c • b) (fun _ => b ⬝ᵥ x + d * c) := by
  classical
  ext i
  cases i with
  | inl i =>
      simp [border, mulVec, dotProduct, replicateCol, of_apply, mul_comm]
  | inr u =>
      simp [border, mulVec, dotProduct, replicateRow, of_apply]

/-- The `1×1` Schur complement of the corner has single entry `S = d - bᵀA⁻¹b`. -/
lemma schur_entry (A : Matrix m m ℝ) [Invertible A] (b : m → ℝ) (d : ℝ) (u v : Unit) :
    ((of fun _ _ => d : Matrix Unit Unit ℝ)
        - replicateRow Unit b * ⅟A * replicateCol Unit b) u v
      = d - b ⬝ᵥ (⅟A *ᵥ b) := by
  classical
  simp [Matrix.mul_apply, dotProduct, mulVec, replicateRow, replicateCol, of_apply,
    Finset.sum_mul, Finset.mul_sum, mul_assoc]
  rw [Finset.sum_comm]

lemma det_border (A : Matrix m m ℝ) [Invertible A] (b : m → ℝ) (d : ℝ) :
    (border A b d).det = A.det * (d - b ⬝ᵥ (⅟A *ᵥ b)) := by
  rw [border, det_fromBlocks₁₁]
  congr 1
  rw [det_unique, schur_entry]

/-- If `M *ᵥ x = Y` and `M` is invertible then `x` is `M⁻¹ *ᵥ Y`. -/
lemma inv_mulVec_eq_of_mulVec_eq {n : Type*} [Fintype n] [DecidableEq n]
    {M : Matrix n n ℝ} (hM : IsUnit M.det) {x Y : n → ℝ} (h : M *ᵥ x = Y) :
    M⁻¹ *ᵥ Y = x := by
  rw [← h, mulVec_mulVec, nonsing_inv_mul _ hM, one_mulVec]

theorem deflection_border_general (A : Matrix m m ℝ) [Invertible A] (b : m → ℝ) (d : ℝ)
    (y : m → ℝ) (η : ℝ) (hS : d - b ⬝ᵥ (⅟A *ᵥ b) ≠ 0) :
    (Sum.elim y (fun _ => η)) ⬝ᵥ ((border A b d)⁻¹ *ᵥ Sum.elim y (fun _ => η))
      = y ⬝ᵥ (⅟A *ᵥ y)
        + (η - b ⬝ᵥ (⅟A *ᵥ y)) * (η - y ⬝ᵥ (⅟A *ᵥ b)) / (d - b ⬝ᵥ (⅟A *ᵥ b)) := by
  classical
  have hdet : IsUnit (border A b d).det := by
    rw [det_border]
    exact (isUnit_det_of_invertible A).mul (Ne.isUnit hS)
  have hAy : A *ᵥ (⅟A *ᵥ y) = y := by rw [mulVec_mulVec, mul_invOf_self, one_mulVec]
  have hAb : A *ᵥ (⅟A *ᵥ b) = b := by rw [mulVec_mulVec, mul_invOf_self, one_mulVec]
  have hmul : border A b d *ᵥ
      Sum.elim (⅟A *ᵥ y - ((η - b ⬝ᵥ (⅟A *ᵥ y)) / (d - b ⬝ᵥ (⅟A *ᵥ b))) • (⅟A *ᵥ b))
        (fun _ => (η - b ⬝ᵥ (⅟A *ᵥ y)) / (d - b ⬝ᵥ (⅟A *ᵥ b)))
      = Sum.elim y (fun _ => η) := by
    rw [border_mulVec]
    congr 1
    · rw [mulVec_sub, mulVec_smul, hAy, hAb]
      abel
    · funext _
      rw [dotProduct_sub, dotProduct_smul, smul_eq_mul]
      field_simp
      ring
  rw [inv_mulVec_eq_of_mulVec_eq hdet hmul, dotProduct, Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, Finset.univ_unique, Finset.sum_singleton]
  rw [← dotProduct, dotProduct_sub, dotProduct_smul, smul_eq_mul]
  field_simp
  ring

/-- The inverse of a symmetric matrix is symmetric. -/
lemma invOf_isSymm (A : Matrix m m ℝ) [Invertible A] (hA : A.IsSymm) : (⅟A).IsSymm := by
  show (⅟A)ᵀ = ⅟A
  rw [invOf_eq_nonsing_inv, transpose_nonsing_inv, hA.eq]

/-- For a symmetric metric the two bilinear pairings agree: `bᵀA⁻¹y = yᵀA⁻¹b`. -/
lemma dotProduct_invOf_comm (A : Matrix m m ℝ) [Invertible A] (hA : A.IsSymm) (u v : m → ℝ) :
    u ⬝ᵥ (⅟A *ᵥ v) = v ⬝ᵥ (⅟A *ᵥ u) := by
  rw [dotProduct_mulVec, ← mulVec_transpose, (invOf_isSymm A hA).eq, dotProduct_comm]

/-- **Bordered-metric (Schur) identity for the deflection, L4.**
Adding one column `ψ` (cross-covariances `b`, own variance `d`) to a dictionary with symmetric
invertible metric `A` and target `(y, η)` gives
`J = yᵀA⁻¹y + (η - bᵀA⁻¹y)² / S` with `S = d - bᵀA⁻¹b`. -/
theorem deflection_border (A : Matrix m m ℝ) [Invertible A] (hA : A.IsSymm) (b : m → ℝ) (d : ℝ)
    (y : m → ℝ) (η : ℝ) (hS : d - b ⬝ᵥ (⅟A *ᵥ b) ≠ 0) :
    (Sum.elim y (fun _ => η)) ⬝ᵥ ((border A b d)⁻¹ *ᵥ Sum.elim y (fun _ => η))
      = y ⬝ᵥ (⅟A *ᵥ y) + (η - b ⬝ᵥ (⅟A *ᵥ y)) ^ 2 / (d - b ⬝ᵥ (⅟A *ᵥ b)) := by
  rw [deflection_border_general A b d y η hS, dotProduct_invOf_comm A hA y b]
  ring

/-- **Monotonicity, L4.** With a positive Schur complement `S > 0`, adding a column never
decreases the deflection. -/
theorem deflection_border_mono (A : Matrix m m ℝ) [Invertible A] (hA : A.IsSymm) (b : m → ℝ)
    (d : ℝ) (y : m → ℝ) (η : ℝ) (hS : 0 < d - b ⬝ᵥ (⅟A *ᵥ b)) :
    y ⬝ᵥ (⅟A *ᵥ y)
      ≤ (Sum.elim y (fun _ => η)) ⬝ᵥ ((border A b d)⁻¹ *ᵥ Sum.elim y (fun _ => η)) := by
  rw [deflection_border A hA b d y η hS.ne']
  have : 0 ≤ (η - b ⬝ᵥ (⅟A *ᵥ y)) ^ 2 / (d - b ⬝ᵥ (⅟A *ᵥ b)) := div_nonneg (sq_nonneg _) hS.le
  linarith

/-- Strict gain: the deflection increases when the new column carries residual target
mass. Only this direction is formalised; the converse is immediate from the same identity. -/
theorem deflection_border_strict (A : Matrix m m ℝ) [Invertible A] (hA : A.IsSymm) (b : m → ℝ)
    (d : ℝ) (y : m → ℝ) (η : ℝ) (hS : 0 < d - b ⬝ᵥ (⅟A *ᵥ b))
    (hr : η - b ⬝ᵥ (⅟A *ᵥ y) ≠ 0) :
    y ⬝ᵥ (⅟A *ᵥ y)
      < (Sum.elim y (fun _ => η)) ⬝ᵥ ((border A b d)⁻¹ *ᵥ Sum.elim y (fun _ => η)) := by
  rw [deflection_border A hA b d y η hS.ne']
  have : 0 < (η - b ⬝ᵥ (⅟A *ᵥ y)) ^ 2 / (d - b ⬝ᵥ (⅟A *ᵥ b)) :=
    div_pos (by positivity) hS
  linarith

end MCP
