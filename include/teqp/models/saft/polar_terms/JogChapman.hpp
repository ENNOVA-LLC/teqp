#pragma once
#include "types.hpp"
#include "teqp/math/pow_templates.hpp"
// Jog & Chapman never developed a quadrupolar contribution; for any
// quadrupolar / cross dipolar-quadrupolar physics we reuse the Gross
// formulations directly.
#include "teqp/models/saft/polar_terms/GrossVrabec.hpp"

namespace teqp::saft::polar_terms::JogChapman{

/**
 The Jog & Chapman dipolar contribution for PC-SAFT.

 Implementation follows Dominik et al. (2005), IECR 44(17), with the closed-form
 reference-fluid integrals from the original Jog & Chapman (1999), Mol. Phys.
 97(3) -- specifically the Padé-resummed I2 and I3 in Eqs. 26-27 of JC 1999.

 Two simplifications relative to Gross-Vrabec (GV):

 1. The pair correlation kernels I2 and I3 are evaluated at the mole-fraction
    averaged segment number m_bar = sum_i x_i m_i instead of the per-pair
    geometric m_ij = sqrt(m_i m_j). Dominik Eqs. 5-7. As a result, I2 and I3
    are scalar functions of (eta, m_bar) only -- they pull *out* of the i,j
    (and i,j,k) sums.

 2. The segment diameter d_ij is taken as the geometric mean sqrt(d_i d_j)
    instead of GV's arithmetic (d_i + d_j)/2. Combined with simplification 1,
    this lets the A2 double sum factor as ( sum_i a_i )^2 and the A3 triple
    sum factor as ( sum_i b_i )^3 -- collapsing O(N^2) and O(N^3) work to O(N).

 No m_ij clamp is needed (GV clamps to 2.0 to avoid bad chain-correction
 extrapolation; JC's kernel has no such pathology because m_bar is the
 composition average and no piecewise correction is involved).
 */

/// JC 1999 Eq. 26: Padé-resummed reference-fluid I2(rho*).
/// rho_star = 6 eta / pi for the PC-SAFT diameter d_i. The published
/// coefficients are for the pure reference fluid.
template <typename EtaType>
auto get_I2(const EtaType& eta) {
    using std::common_type_t;
    auto rho_star = forceeval(6.0 * eta / static_cast<double>(EIGEN_PI));
    auto num = 1.0 - 0.3618 * rho_star
                   - 0.3205 * pow(rho_star, 2)
                   + 0.1078 * pow(rho_star, 3);
    auto den = pow(1.0 - 0.5236 * rho_star, 2);
    return forceeval(num / den);
}

/// JC 1999 Eq. 27: Padé-resummed reference-fluid I3(rho*).
template <typename EtaType>
auto get_I3(const EtaType& eta) {
    auto rho_star = forceeval(6.0 * eta / static_cast<double>(EIGEN_PI));
    auto num = 1.0 + 0.62378 * rho_star - 0.11658 * pow(rho_star, 2);
    auto den = 1.0 - 0.59056 * rho_star + 0.20059 * pow(rho_star, 2);
    return forceeval(num / den);
}

/**
 \brief Dipolar contribution from Jog & Chapman / Dominik et al.
![1780063830775](image/JogChapman/1780063830775.png)![1780063854560](image/JogChapman/1780063854560.png)
 Inputs match the published JC formulation directly:

 - ``m`` segment number per species (dimensionless).
 - ``xp`` fraction of dipolar segments on the chain (dimensionless).
 - ``mu_squared_SI`` molecular dipole moment squared, in SI units (C^2 m^2).
   To convert from Debye^2: multiply by ``(3.33564e-30)^2``. Marshall's
   polar strength ``alpha_p = m x_p mu^2`` (in D^2) can be supplied as
   ``mu_squared_SI = alpha_p_D2 * (3.33564e-30)^2 / (m * xp)`` if your
   data is in that form.
 - ``sigmaij_rule`` (default ``geometric``) chooses the d_ij combining
   rule. ``geometric`` (d_ij = sqrt(d_i d_j)) is the AM 2024 / Marshall
   convention and lets the i,j and i,j,k sums collapse to O(N).
   ``arithmetic`` (d_ij = (d_i + d_j)/2) is the more general Lorentz-
   Berthelot rule; it evaluates the full O(N^2) / O(N^3) sums.

 At every ``eval()`` call the caller supplies ``d_Angstrom``, the per-
 segment diameter used in the polar pair integral. This class applies
 no transformation to that diameter -- the caller chooses what to pass:

 - PC-SAFT-with-JC: d_i(T) = sigma_i * (1 - 0.12 exp(-3 eps_i/(k_B T)))
   (the Chen-Kreglewski hard-chain diameter, recomputed per call).
 - Cubic-SAFT (Abutaqiya-Marshall 2024 SP-SRK): d_i = (3 b_i / (2 pi N_A))^(1/3)
   (the cubic-derived diameter, temperature-independent).

 The kernel formula is (Marshall-Bokis 2019 / AM 2024 / Dominik 2005):

     a_2^DD = -(2 pi / 9) (1/(4 pi eps_0))^2 * rho / (k_B T)^2
              * Sum_ij x_i x_j m_i m_j x_p,i x_p,j mu^2_i mu^2_j / d_ij^3 * I_2(eta)

 With geometric d_ij the i,j sum collapses to a single sum squared;
 with arithmetic d_ij it does not factor and the full pair sum runs.
 */
class DipolarContributionJogChapman {
private:
    const Eigen::ArrayXd m, xp, mu_squared_SI;
    const SigmaijRule sigmaij_rule;
    // Physical constants (SI)
    static constexpr double FOUR_PI_EPS0 = 1.11265005605362e-10;  // 4*pi*eps_0 [C^2/(N m^2)]
    static constexpr double K_B = 1.380649e-23;                    // Boltzmann [J/K]

    /// d_ij in meters, given d_i and d_j in meters and the active rule.
    template <typename DType>
    auto pair_d(const DType& d_i, const DType& d_j) const {
        if (sigmaij_rule == SigmaijRule::geometric) {
            return forceeval(sqrt(d_i * d_j));
        }
        return forceeval(0.5 * (d_i + d_j));
    }

public:
    const bool has_a_polar;

    /// Construct with raw per-species inputs in SI / dimensionless units.
    /// ``mu_squared_SI[i]`` is the molecule's dipole moment squared in C^2 m^2.
    DipolarContributionJogChapman(
        const Eigen::ArrayX<double>& m,
        const Eigen::ArrayX<double>& xp,
        const Eigen::ArrayX<double>& mu_squared_SI,
        SigmaijRule sigmaij_rule = SigmaijRule::geometric)
    : m(m), xp(xp), mu_squared_SI(mu_squared_SI),
      sigmaij_rule(sigmaij_rule),
      has_a_polar((mu_squared_SI * xp).cwiseAbs().sum() > 0)
    {
        if (m.size() != mu_squared_SI.size()) {
            throw teqp::InvalidArgument("bad size of mu_squared_SI");
        }
        if (m.size() != xp.size()) {
            throw teqp::InvalidArgument("bad size of xp");
        }
    }
    DipolarContributionJogChapman& operator=(const DipolarContributionJogChapman&) = delete;

    /// A2 contribution. Dominik 2005 Eq. 3 / Marshall-Bokis 2019.
    ///
    /// Geometric d_ij path (default): the i,j double sum collapses to
    ///     S = Sum_i x_i m_i x_p,i mu^2_i / d_i^{3/2}
    ///     a_2 = -(2 pi / 9) / (4 pi eps_0)^2 * rho_N / (k_B T)^2 * I_2(eta) * S^2
    /// where rho_N is the number density in particles/m^3.
    ///
    /// Arithmetic d_ij path: full O(N^2) pair sum,
    ///     a_2 = -(2 pi / 9) / (4 pi eps_0)^2 * rho_N / (k_B T)^2 * I_2(eta)
    ///           * Sum_ij x_i x_j m_i m_j x_p,i x_p,j mu^2_i mu^2_j / d_ij^3
    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType, typename DVecType>
    auto get_alpha2DD(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta,
                       const VecType& mole_fractions, const DVecType& d_Angstrom) const {
        const auto& x = mole_fractions;
        const auto N = mole_fractions.size();

        using sum_t = std::common_type_t<TTYPE, RhoType, EtaType,
                                          decltype(mole_fractions[0]), decltype(d_Angstrom[0])>;
        auto I2 = get_I2(eta);
        auto rho_N_m3 = rhoN_A3 * 1e30;
        const double prefactor = -2.0 * static_cast<double>(EIGEN_PI) / 9.0
                                  / (FOUR_PI_EPS0 * FOUR_PI_EPS0);

        if (sigmaij_rule == SigmaijRule::geometric) {
            // Collapsed O(N) form. S = Sum_i x_i m_i x_p,i mu^2_i / d_i^{3/2}
            sum_t S = 0.0;
            for (Eigen::Index i = 0; i < N; ++i) {
                if (xp[i] * mu_squared_SI[i] == 0) continue;
                auto d_m = d_Angstrom[i] * 1e-10;       // Angstrom -> meters
                S += x[i] * m[i] * xp[i] * mu_squared_SI[i] / pow(d_m, 1.5);
            }
            return forceeval(prefactor * rho_N_m3 / (K_B * K_B * T * T) * I2 * S * S);
        } else {
            // Arithmetic d_ij: full O(N^2) pair sum, no collapse.
            sum_t summer = 0.0;
            for (Eigen::Index i = 0; i < N; ++i) {
                if (xp[i] * mu_squared_SI[i] == 0) continue;
                auto d_i_m = d_Angstrom[i] * 1e-10;
                for (Eigen::Index j = 0; j < N; ++j) {
                    if (xp[j] * mu_squared_SI[j] == 0) continue;
                    auto d_j_m = d_Angstrom[j] * 1e-10;
                    auto d_ij = pair_d(d_i_m, d_j_m);
                    summer += x[i] * x[j] * m[i] * m[j] * xp[i] * xp[j]
                              * mu_squared_SI[i] * mu_squared_SI[j]
                              / (d_ij * d_ij * d_ij);
                }
            }
            return forceeval(prefactor * rho_N_m3 / (K_B * K_B * T * T) * I2 * summer);
        }
    }

    /// A3 contribution. Dominik 2005 Eq. 4.
    ///
    /// Geometric d_ij path: the i,j,k triple sum collapses to a single
    /// sum cubed,
    ///     S = Sum_i x_i m_i x_p,i mu^2_i / d_i
    ///     a_3 = -(5 pi^2 / 162) / (4 pi eps_0)^3 * rho_N^2 / (k_B T)^3 * I_3(eta) * S^3
    ///
    /// Arithmetic d_ij path: full O(N^3) triple sum.
    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType, typename DVecType>
    auto get_alpha3DD(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta,
                       const VecType& mole_fractions, const DVecType& d_Angstrom) const {
        const auto& x = mole_fractions;
        const auto N = mole_fractions.size();

        using sum_t = std::common_type_t<TTYPE, RhoType, EtaType,
                                          decltype(mole_fractions[0]), decltype(d_Angstrom[0])>;
        auto I3 = get_I3(eta);
        auto rho_N_m3 = rhoN_A3 * 1e30;
        const double pi_d = static_cast<double>(EIGEN_PI);
        const double prefactor = -5.0 * pi_d * pi_d / 162.0
                                  / (FOUR_PI_EPS0 * FOUR_PI_EPS0 * FOUR_PI_EPS0);

        if (sigmaij_rule == SigmaijRule::geometric) {
            sum_t S = 0.0;
            for (Eigen::Index i = 0; i < N; ++i) {
                if (xp[i] * mu_squared_SI[i] == 0) continue;
                auto d_m = d_Angstrom[i] * 1e-10;
                S += x[i] * m[i] * xp[i] * mu_squared_SI[i] / d_m;
            }
            return forceeval(prefactor * POW2(rho_N_m3) / (K_B * K_B * K_B * T * T * T)
                              * I3 * S * S * S);
        } else {
            // Arithmetic d_ij: full O(N^3) triple sum, no collapse.
            sum_t summer = 0.0;
            for (Eigen::Index i = 0; i < N; ++i) {
                if (xp[i] * mu_squared_SI[i] == 0) continue;
                auto d_i_m = d_Angstrom[i] * 1e-10;
                for (Eigen::Index j = 0; j < N; ++j) {
                    if (xp[j] * mu_squared_SI[j] == 0) continue;
                    auto d_j_m = d_Angstrom[j] * 1e-10;
                    auto d_ij = pair_d(d_i_m, d_j_m);
                    for (Eigen::Index k = 0; k < N; ++k) {
                        if (xp[k] * mu_squared_SI[k] == 0) continue;
                        auto d_k_m = d_Angstrom[k] * 1e-10;
                        auto d_ik = pair_d(d_i_m, d_k_m);
                        auto d_jk = pair_d(d_j_m, d_k_m);
                        summer += x[i] * x[j] * x[k]
                                  * m[i] * m[j] * m[k]
                                  * xp[i] * xp[j] * xp[k]
                                  * mu_squared_SI[i] * mu_squared_SI[j] * mu_squared_SI[k]
                                  / (d_ij * d_ik * d_jk);
                    }
                }
            }
            return forceeval(prefactor * POW2(rho_N_m3) / (K_B * K_B * K_B * T * T * T)
                              * I3 * summer);
        }
    }

    /// Padé-resummed dipolar contribution: alpha = alpha2 / (1 - alpha3/alpha2).
    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType, typename DVecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta,
              const VecType& mole_fractions, const DVecType& d_Angstrom) const {
        auto alpha2 = get_alpha2DD(T, rho_A3, eta, mole_fractions, d_Angstrom);
        auto alpha3 = get_alpha3DD(T, rho_A3, eta, mole_fractions, d_Angstrom);
        auto alpha = forceeval(alpha2 / (1.0 - alpha3 / alpha2));

        using alpha2_t = decltype(alpha2);
        using alpha3_t = decltype(alpha3);
        using alpha_t  = decltype(alpha);
        struct DipolarContributionTerms {
            alpha2_t alpha2;
            alpha3_t alpha3;
            alpha_t  alpha;
        };
        return DipolarContributionTerms{alpha2, alpha3, alpha};
    }
};

/**
 \brief Aggregator analogous to MultipolarContributionGrossVrabec.

 Dipolar physics is JC; quadrupolar and cross dipolar-quadrupolar physics is
 borrowed from GV. The structural parallel with the GV aggregator is
 deliberate -- callers of polar_terms.hpp's std::variant should not have to
 know which branch they hold.
 */
template <typename type>
struct MultipolarContributionJogChapmanTerms {
    type alpha2DD;
    type alpha3DD;
    type alphaDD;
    type alpha2QQ;
    type alpha3QQ;
    type alphaQQ;
    type alpha2DQ;
    type alpha3DQ;
    type alphaDQ;
    type alpha;
};

class MultipolarContributionJogChapman {
public:
    static constexpr multipolar_argument_spec arg_spec = multipolar_argument_spec::TK_rhoNA3_packingfraction_molefractions;

    const std::optional<DipolarContributionJogChapman> di;
    // Reuse the GV quadrupolar and cross-DQ classes as-is; they are
    // pair-resolved (O(N^2)/O(N^3)) and so do *not* benefit from the
    // JC mean-field-kernel collapse. Active only when Q-moments are
    // provided alongside the JC dipole parameters.
    const std::optional<GrossVrabec::QuadrupolarContributionGross> quad;
    const std::optional<GrossVrabec::DipolarQuadrupolarContributionVrabecGross> diquad;

    /// Construct the JC aggregator.
    ///
    /// JC dipolar branch consumes (m, xp, mu_squared_SI) directly --
    /// see DipolarContributionJogChapman for unit conventions.
    /// GV quadrupolar and cross-DQ branches still consume
    /// (m, sigma_Angstrom, epsilon_over_k, ...) because GV's published
    /// kernel uses those parameters explicitly. The cross-DQ class also
    /// needs mu_star^2 in GV's reduced form, which the caller must
    /// supply alongside the raw mu^2 if the cross term is wanted; if
    /// only the JC dipolar branch is needed, pass mustar2_GV = 0 to
    /// disable the cross-DQ branch.
    MultipolarContributionJogChapman(
        const Eigen::ArrayX<double>& m,
        const Eigen::ArrayX<double>& sigma_Angstrom,
        const Eigen::ArrayX<double>& epsilon_over_k,
        const Eigen::ArrayX<double>& mu_squared_SI,
        const Eigen::ArrayX<double>& xp,
        const Eigen::ArrayX<double>& Qstar2,
        const Eigen::ArrayX<double>& nQ,
        const Eigen::ArrayX<double>& mustar2_GV_for_cross_DQ,
        SigmaijRule sigmaij_rule = SigmaijRule::geometric)
    : di((((xp * mu_squared_SI > 0).cast<int>().sum() > 0)
              ? decltype(di)(DipolarContributionJogChapman(m, xp, mu_squared_SI, sigmaij_rule))
              : std::nullopt)),
      quad((((nQ * Qstar2 > 0).cast<int>().sum() > 0)
              ? decltype(quad)(GrossVrabec::QuadrupolarContributionGross(m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ, sigmaij_rule))
              : std::nullopt)),
      // Cross DQ requires *both* dipole and quadrupole. The GV cross-DQ class
      // expects an nmu input (GV's reduced dipole convention), which JC doesn't
      // carry natively -- the caller must supply mustar2_GV_for_cross_DQ along
      // with a derived nmu = m * xp so the GV term sees a consistent count.
      diquad((di && quad && (mustar2_GV_for_cross_DQ * xp > 0).cast<int>().sum() > 0)
              ? decltype(diquad)(GrossVrabec::DipolarQuadrupolarContributionVrabecGross(
                    m, sigma_Angstrom, epsilon_over_k, mustar2_GV_for_cross_DQ,
                    (m * xp).eval(), Qstar2, nQ, sigmaij_rule))
              : std::nullopt)
    {}

    /// Evaluate the JC dipolar + GV quadrupolar + GV cross-DQ contributions.
    /// ``d_Angstrom`` is the per-eval diameter vector consumed by the JC
    /// dipolar branch only; the GV quadrupolar and cross-DQ branches use
    /// the construction-time sigma directly (their published kernel has
    /// no segment-diameter temperature correction).
    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType, typename DVecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta,
              const VecType& mole_fractions, const DVecType& d_Angstrom) const {
        using type = std::common_type_t<TTYPE, RhoType, EtaType,
                                         decltype(mole_fractions[0]), decltype(d_Angstrom[0])>;
        type alpha2DD = 0.0, alpha3DD = 0.0, alphaDD = 0.0;
        type alpha2QQ = 0.0, alpha3QQ = 0.0, alphaQQ = 0.0;
        type alpha2DQ = 0.0, alpha3DQ = 0.0, alphaDQ = 0.0;
        if (di && di.value().has_a_polar) {
            alpha2DD = di.value().get_alpha2DD(T, rho_A3, eta, mole_fractions, d_Angstrom);
            alpha3DD = di.value().get_alpha3DD(T, rho_A3, eta, mole_fractions, d_Angstrom);
            alphaDD  = forceeval(alpha2DD / (1.0 - alpha3DD / alpha2DD));
        }
        if (quad && quad.value().has_a_polar) {
            alpha2QQ = quad.value().get_alpha2QQ(T, rho_A3, eta, mole_fractions);
            alpha3QQ = quad.value().get_alpha3QQ(T, rho_A3, eta, mole_fractions);
            alphaQQ  = forceeval(alpha2QQ / (1.0 - alpha3QQ / alpha2QQ));
        }
        if (diquad) {
            alpha2DQ = diquad.value().get_alpha2DQ(T, rho_A3, eta, mole_fractions);
            alpha3DQ = diquad.value().get_alpha3DQ(T, rho_A3, eta, mole_fractions);
            alphaDQ  = forceeval(alpha2DQ / (1.0 - alpha3DQ / alpha2DQ));
        }
        auto alpha = forceeval(alphaDD + alphaQQ + alphaDQ);
        return MultipolarContributionJogChapmanTerms<type>{
            alpha2DD, alpha3DD, alphaDD,
            alpha2QQ, alpha3QQ, alphaQQ,
            alpha2DQ, alpha3DQ, alphaDQ,
            alpha};
    }
};

} // namespace teqp::saft::polar_terms::JogChapman
