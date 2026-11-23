import Mathlib
import MCP.Schur

set_option autoImplicit false

/-!
# Deterministic cores of the reframed H1 (paper/reframe/main.tex)

One declaration per deterministic lemma of the theory note. The probabilistic statements
(point-process limits, stable window limits) are not formalized; see the Theorem map in
PIPELINE.md. Every expression is written inline (no `def`), so the frozen statement text
fixes what is proved.
-/

open Matrix Real

namespace MCP.Reframe

variable {m : Type*} [Fintype m] [DecidableEq m]

/-- `lem:schur` — bordered solve: top coefficient and admissible block of `F k = y`. -/
theorem bordered_solve (A : Matrix m m ℝ) [Invertible A] (f : m → ℝ) (d : ℝ)
    (kL yL : m → ℝ) (kt yt : ℝ)
    (h : border A f d *ᵥ Sum.elim kL (fun _ => kt) = Sum.elim yL (fun _ => yt))
    (hS : d - f ⬝ᵥ (⅟A *ᵥ f) ≠ 0) :
    kt = (yt - f ⬝ᵥ (⅟A *ᵥ yL)) / (d - f ⬝ᵥ (⅟A *ᵥ f)) ∧
      kL = ⅟A *ᵥ (yL - kt • f) := by
  rw [border_mulVec] at h
  have hL : A *ᵥ kL = yL - kt • f := by
    funext i
    have := congrFun h (Sum.inl i)
    simp only [Sum.elim_inl, Pi.add_apply, Pi.smul_apply] at this
    simp only [Pi.sub_apply, Pi.smul_apply]
    linarith
  have ht : f ⬝ᵥ kL + d * kt = yt := by simpa using congrFun h (Sum.inr ())
  have hk : kL = ⅟A *ᵥ (yL - kt • f) := by
    rw [← hL, mulVec_mulVec, invOf_mul_self, one_mulVec]
  refine ⟨?_, hk⟩
  rw [eq_div_iff hS]
  rw [hk, mulVec_sub, mulVec_smul, dotProduct_sub, dotProduct_smul, smul_eq_mul] at ht
  linear_combination ht

theorem bordered_solve_witness :
    ∃ (f : Fin 1 → ℝ) (d : ℝ) (kL yL : Fin 1 → ℝ) (kt yt : ℝ),
      border (1 : Matrix (Fin 1) (Fin 1) ℝ) f d *ᵥ Sum.elim kL (fun _ => kt)
          = Sum.elim yL (fun _ => yt) ∧
        d - f ⬝ᵥ ((invertibleOne : Invertible (1 : Matrix (Fin 1) (Fin 1) ℝ)).invOf *ᵥ f) ≠ 0 := by
  refine ⟨0, 1, 0, 0, 0, 0, ?_, ?_⟩
  · rw [border_mulVec]
    funext i
    cases i <;> simp
  · simp

/-- `lem:parity`, same parity — even·even and odd·odd are `|x|^(a+b) > 0` off the origin. -/
theorem same_parity_product (a b x : ℝ) (hx : x ≠ 0) :
    |x| ^ a * |x| ^ b = |x| ^ (a + b) ∧ 0 < |x| ^ a * |x| ^ b ∧
      (Real.sign x * |x| ^ a) * (Real.sign x * |x| ^ b) = |x| ^ (a + b) ∧
      0 < (Real.sign x * |x| ^ a) * (Real.sign x * |x| ^ b) := by
  have hpos : 0 < |x| := abs_pos.mpr hx
  have hadd : |x| ^ a * |x| ^ b = |x| ^ (a + b) := (Real.rpow_add hpos a b).symm
  have hss : Real.sign x * Real.sign x = 1 := by
    rcases lt_or_gt_of_ne hx with h | h
    · rw [Real.sign_of_neg h]; norm_num
    · rw [Real.sign_of_pos h]; norm_num
  have hodd : (Real.sign x * |x| ^ a) * (Real.sign x * |x| ^ b) = |x| ^ (a + b) := by
    rw [← hadd]; linear_combination (|x| ^ a * |x| ^ b) * hss
  have hp : 0 < |x| ^ (a + b) := Real.rpow_pos_of_pos hpos _
  exact ⟨hadd, by rw [hadd]; exact hp, hodd, by rw [hodd]; exact hp⟩

theorem same_parity_product_witness : ∃ (a b x : ℝ), x ≠ 0 := by
  exact ⟨0, 0, 1, one_ne_zero⟩

/-- `lem:parity`, opposite parity — even·odd is an odd function. -/
theorem opposite_parity_product (a b x : ℝ) :
    |(-x)| ^ a * (Real.sign (-x) * |(-x)| ^ b) = -(|x| ^ a * (Real.sign x * |x| ^ b)) := by
  simp only [abs_neg, Real.sign_neg]
  ring

/-- `lem:possum` — a same-parity sum over points not all at the origin is positive, so
`-c · sum / D` has the sign of `-c`. Parity `p = true` is odd, `false` is even. -/
theorem same_parity_sum_sign {ι : Type*} (s : Finset ι) (x : ι → ℝ) (p : Bool)
    (a b c D : ℝ) (hx : ∃ k ∈ s, x k ≠ 0) (hc : c ≠ 0) (hD : 0 < D) :
    0 < ∑ k ∈ s, (if p then Real.sign (x k) * |x k| ^ a else |x k| ^ a) *
        (if p then Real.sign (x k) * |x k| ^ b else |x k| ^ b) ∧
      Real.sign (-c * (∑ k ∈ s, (if p then Real.sign (x k) * |x k| ^ a else |x k| ^ a) *
        (if p then Real.sign (x k) * |x k| ^ b else |x k| ^ b)) / D) = -Real.sign c := by
  have hnn : ∀ y : ℝ, 0 ≤ (if p then Real.sign y * |y| ^ a else |y| ^ a) *
      (if p then Real.sign y * |y| ^ b else |y| ^ b) := by
    intro y
    by_cases hy : y = 0
    · subst hy
      cases p
      · exact mul_nonneg (Real.rpow_nonneg (abs_nonneg _) _) (Real.rpow_nonneg (abs_nonneg _) _)
      · show 0 ≤ (Real.sign 0 * |(0:ℝ)| ^ a) * (Real.sign 0 * |(0:ℝ)| ^ b)
        simp [Real.sign_zero]
    · have := same_parity_product a b y hy
      cases p
      · exact this.2.1.le
      · exact this.2.2.2.le
  have hpos : ∀ y : ℝ, y ≠ 0 → 0 < (if p then Real.sign y * |y| ^ a else |y| ^ a) *
      (if p then Real.sign y * |y| ^ b else |y| ^ b) := by
    intro y hy
    have := same_parity_product a b y hy
    cases p
    · exact this.2.1
    · exact this.2.2.2
  obtain ⟨k0, hk0, hxk0⟩ := hx
  have hsgn : ∀ S : ℝ, 0 < S → Real.sign (-c * S / D) = -Real.sign c := by
    intro S hS
    rcases lt_or_gt_of_ne hc with h | h
    · have : 0 < -c * S / D := div_pos (mul_pos (neg_pos.mpr h) hS) hD
      rw [Real.sign_of_pos this, Real.sign_of_neg h]; norm_num
    · have : -c * S / D < 0 := div_neg_of_neg_of_pos (mul_neg_of_neg_of_pos (neg_neg_of_pos h) hS) hD
      rw [Real.sign_of_neg this, Real.sign_of_pos h]
  have hS := Finset.sum_pos' (fun k _ => hnn (x k)) ⟨k0, hk0, hpos (x k0) hxk0⟩
  exact ⟨hS, hsgn _ hS⟩

theorem same_parity_sum_sign_witness :
    ∃ (s : Finset (Fin 1)) (x : Fin 1 → ℝ) (c D : ℝ),
      (∃ k ∈ s, x k ≠ 0) ∧ c ≠ 0 ∧ 0 < D := by
  exact ⟨{0}, fun _ => 1, 1, 1, ⟨0, Finset.mem_singleton_self 0, one_ne_zero⟩, one_ne_zero, one_pos⟩

/-- `lem:rho` — Zolotarev's positivity parameter
`ρ(γ, β) = 1/2 + arctan(β tan(πγ/2)) / (πγ)` for `0 < γ < 1`. -/
theorem rho_props (γ β : ℝ) (hγ0 : 0 < γ) (hγ1 : γ < 1) :
    (1 / 2 + arctan (β * tan (π * γ / 2)) / (π * γ)) +
        (1 / 2 + arctan ((-β) * tan (π * γ / 2)) / (π * γ)) = 1 ∧
      (1 / 2 + arctan (β * tan (π * γ / 2)) / (π * γ) < 1 / 2 ↔ β < 0) ∧
      1 / 2 + arctan (0 * tan (π * γ / 2)) / (π * γ) = 1 / 2 := by
  have hpg : 0 < π * γ := mul_pos Real.pi_pos hγ0
  have ht : 0 < tan (π * γ / 2) :=
    Real.tan_pos_of_pos_of_lt_pi_div_two (half_pos hpg) (by nlinarith [Real.pi_pos])
  refine ⟨by rw [neg_mul, Real.arctan_neg]; ring, ?_, by simp⟩
  constructor
  · intro h
    have h1 : arctan (β * tan (π * γ / 2)) / (π * γ) < 0 := by linarith
    have h2 : arctan (β * tan (π * γ / 2)) < 0 := by
      by_contra hc; push_neg at hc; have := div_nonneg hc hpg.le; linarith
    have h3 : β * tan (π * γ / 2) < 0 := by
      by_contra hc; push_neg at hc
      have := Real.arctan_strictMono.monotone hc; rw [Real.arctan_zero] at this; linarith
    by_contra hb; push_neg at hb; have := mul_nonneg hb ht.le; linarith
  · intro hb
    have h3 : β * tan (π * γ / 2) < 0 := mul_neg_of_neg_of_pos hb ht
    have h2 : arctan (β * tan (π * γ / 2)) < 0 := by
      have := Real.arctan_strictMono h3; rwa [Real.arctan_zero] at this
    have := div_neg_of_neg_of_pos h2 hpg; linarith

theorem rho_props_witness : ∃ γ : ℝ, 0 < γ ∧ γ < 1 := by
  exact ⟨1 / 2, by norm_num, by norm_num⟩

/-- `lem:balance` — the crossover `N* = (μ/σ)^(α/(a-α)) κ^(-a/(a-α))` balances the heavy
scale `σ (κN)^(a/α)` against the drift `μN`, the ratio is `(N/N*)^((a-α)/α)`, and with
`σ = σ₀ (c₀ n ε)^(-(a-a_L)/α)` the `n`-exponent is `(a-a_L)/(a-α)`. -/
theorem crossover_balance (aS aL αc μ σ κ σ0 M : ℝ) (hα : 0 < αc) (ha : αc < aS)
    (hμ : 0 < μ) (hσ : 0 < σ) (hκ : 0 < κ) (hσ0 : 0 < σ0) (hM : 0 < M) :
    σ * (κ * ((μ / σ) ^ (αc / (aS - αc)) * κ ^ (-(aS / (aS - αc))))) ^ (aS / αc)
        = μ * ((μ / σ) ^ (αc / (aS - αc)) * κ ^ (-(aS / (aS - αc)))) ∧
      (∀ N : ℝ, 0 < N →
        σ * (κ * N) ^ (aS / αc) / (μ * N)
          = (N / ((μ / σ) ^ (αc / (aS - αc)) * κ ^ (-(aS / (aS - αc))))) ^ ((aS - αc) / αc)) ∧
      (μ / (σ0 * M ^ (-((aS - aL) / αc)))) ^ (αc / (aS - αc)) * κ ^ (-(aS / (aS - αc)))
        = (μ / σ0) ^ (αc / (aS - αc)) * κ ^ (-(aS / (aS - αc))) * M ^ ((aS - aL) / (aS - αc)) := by
  have hd : aS - αc ≠ 0 := (sub_pos.mpr ha).ne'
  have hαne : αc ≠ 0 := hα.ne'
  have hp : ∀ x y : ℝ, 0 < x → 0 < x ^ y := fun x y hx => Real.rpow_pos_of_pos hx y
  have hμσ : 0 < μ / σ := div_pos hμ hσ
  have hA := hp (μ / σ) (αc / (aS - αc)) hμσ
  have hB := hp κ (-(aS / (aS - αc))) hκ
  have hNs := mul_pos hA hB
  have lA : Real.log ((μ / σ) ^ (αc / (aS - αc))) = αc / (aS - αc) * (Real.log μ - Real.log σ) := by
    rw [Real.log_rpow hμσ, Real.log_div hμ.ne' hσ.ne']
  have lB : Real.log (κ ^ (-(aS / (aS - αc)))) = -(aS / (aS - αc)) * Real.log κ :=
    Real.log_rpow hκ _
  have lNs : Real.log ((μ / σ) ^ (αc / (aS - αc)) * κ ^ (-(aS / (aS - αc))))
      = αc / (aS - αc) * (Real.log μ - Real.log σ) - aS / (aS - αc) * Real.log κ := by
    rw [Real.log_mul hA.ne' hB.ne', lA, lB]; ring
  refine ⟨?_, ?_, ?_⟩
  · apply Real.log_injOn_pos (Set.mem_Ioi.mpr (mul_pos hσ (hp _ _ (mul_pos hκ hNs))))
      (Set.mem_Ioi.mpr (mul_pos hμ hNs))
    rw [Real.log_mul hσ.ne' (hp _ _ (mul_pos hκ hNs)).ne', Real.log_rpow (mul_pos hκ hNs),
      Real.log_mul hκ.ne' hNs.ne', Real.log_mul hμ.ne' hNs.ne', lNs]
    field_simp
    ring
  · intro N hN
    have hL := mul_pos hσ (hp _ (aS / αc) (mul_pos hκ hN))
    apply Real.log_injOn_pos (Set.mem_Ioi.mpr (div_pos hL (mul_pos hμ hN)))
      (Set.mem_Ioi.mpr (hp _ _ (div_pos hN hNs)))
    rw [Real.log_div hL.ne' (mul_pos hμ hN).ne', Real.log_mul hσ.ne' (hp _ _ (mul_pos hκ hN)).ne',
      Real.log_rpow (mul_pos hκ hN), Real.log_mul hκ.ne' hN.ne', Real.log_mul hμ.ne' hN.ne',
      Real.log_rpow (div_pos hN hNs), Real.log_div hN.ne' hNs.ne', lNs]
    field_simp
    ring
  · have hMr := hp M (-((aS - aL) / αc)) hM
    have hden := mul_pos hσ0 hMr
    have hC := hp (μ / σ0) (αc / (aS - αc)) (div_pos hμ hσ0)
    have hD := hp M ((aS - aL) / (aS - αc)) hM
    apply Real.log_injOn_pos (Set.mem_Ioi.mpr (mul_pos (hp _ _ (div_pos hμ hden)) hB))
      (Set.mem_Ioi.mpr (mul_pos (mul_pos hC hB) hD))
    rw [Real.log_mul (hp _ _ (div_pos hμ hden)).ne' hB.ne', Real.log_rpow (div_pos hμ hden),
      Real.log_div hμ.ne' hden.ne', Real.log_mul hσ0.ne' hMr.ne', Real.log_rpow hM,
      Real.log_mul (mul_pos hC hB).ne' hD.ne', Real.log_mul hC.ne' hB.ne',
      Real.log_rpow (div_pos hμ hσ0), Real.log_div hμ.ne' hσ0.ne', lB, Real.log_rpow hM]
    field_simp
    ring

theorem crossover_balance_witness :
    ∃ (aS αc μ σ κ σ0 M : ℝ), 0 < αc ∧ αc < aS ∧ 0 < μ ∧ 0 < σ ∧ 0 < κ ∧ 0 < σ0 ∧ 0 < M := by
  exact ⟨2, 1, 1, 1, 1, 1, 1, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num⟩

end MCP.Reframe
