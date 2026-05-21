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
 */
class DipolarContributionJogChapman {
private:
    const Eigen::ArrayXd m, sigma_Angstrom, epsilon_over_k, mustar2, xp;
public:
    const bool has_a_polar;

    DipolarContributionJogChapman(
        const Eigen::ArrayX<double>& m,
        const Eigen::ArrayX<double>& sigma_Angstrom,
        const Eigen::ArrayX<double>& epsilon_over_k,
        const Eigen::ArrayX<double>& mustar2,
        const Eigen::ArrayX<double>& xp)
    : m(m), sigma_Angstrom(sigma_Angstrom), epsilon_over_k(epsilon_over_k),
      mustar2(mustar2), xp(xp),
      has_a_polar((mustar2 * xp).cwiseAbs().sum() > 0)
    {
        if (m.size() != mustar2.size()) {
            throw teqp::InvalidArgument("bad size of mustar2");
        }
        if (m.size() != xp.size()) {
            throw teqp::InvalidArgument("bad size of xp");
        }
    }
    DipolarContributionJogChapman& operator=(const DipolarContributionJogChapman&) = delete;

    /// Compute the temperature-dependent Chen-Kreglewski segment diameter
    /// d_i = sigma_i * (1 - 0.12 * exp(-3 eps_i / (k T)))
    /// matching the convention used by PCSAFTHardChainContribution.
    template <typename TTYPE>
    auto get_d(const TTYPE& T) const {
        Eigen::ArrayX<TTYPE> d(m.size());
        for (Eigen::Index i = 0; i < m.size(); ++i) {
            d[i] = sigma_Angstrom[i] * (1.0 - 0.12 * exp(-3.0 * epsilon_over_k[i] / T));
        }
        return d;
    }

    /// A2 contribution. Dominik Eq. 3, with I2 pulled out of the i,j sum
    /// (it depends only on eta and m_bar) and d_ij = sqrt(d_i d_j) so the
    /// d_ij^3 prefactor separates as d_i^{3/2} d_j^{3/2}. The double sum
    /// collapses to ( sum_i a_i )^2 with a_i = x_i m_i x_p,i (mu*_i)^2
    /// (eps_i / kT) sigma_i^3 / d_i^{3/2}.
    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha2DD(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const {
        const auto& x = mole_fractions;
        const auto& sigma = sigma_Angstrom;
        const auto N = mole_fractions.size();
        auto d = get_d(T);

        using sum_t = std::common_type_t<TTYPE, RhoType, EtaType, decltype(mole_fractions[0])>;
        sum_t S = 0.0;
        for (Eigen::Index i = 0; i < N; ++i) {
            if (xp[i] * mustar2[i] == 0) continue;
            S += x[i] * m[i] * xp[i] * mustar2[i] * (epsilon_over_k[i] / T)
                 * POW3(sigma[i]) / pow(d[i], 1.5);
        }
        auto I2 = get_I2(eta);
        // Sign and 2*pi/9 prefactor follow Dominik Eq. 3 (after folding eps/kT
        // into the per-component scalar so the units match GV's (mu*)^2 form).
        return forceeval(-2.0 * static_cast<double>(EIGEN_PI) / 9.0 * rhoN_A3 * I2 * S * S);
    }

    /// A3 contribution. Dominik Eq. 4. With d_ij = sqrt(d_i d_j), the triple
    /// prefactor 1/(d_ij d_ik d_jk) = 1/(d_i d_j d_k) exactly, and the I3
    /// kernel is mole-fraction-averaged, so the triple sum collapses to
    /// ( sum_i b_i )^3 with b_i = x_i m_i x_p,i (mu*_i)^2 (eps_i / kT) sigma_i^3 / d_i.
    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto get_alpha3DD(const TTYPE& T, const RhoType& rhoN_A3, const EtaType& eta, const VecType& mole_fractions) const {
        const auto& x = mole_fractions;
        const auto& sigma = sigma_Angstrom;
        const auto N = mole_fractions.size();
        auto d = get_d(T);

        using sum_t = std::common_type_t<TTYPE, RhoType, EtaType, decltype(mole_fractions[0])>;
        sum_t S = 0.0;
        for (Eigen::Index i = 0; i < N; ++i) {
            if (xp[i] * mustar2[i] == 0) continue;
            S += x[i] * m[i] * xp[i] * mustar2[i] * (epsilon_over_k[i] / T)
                 * POW3(sigma[i]) / d[i];
        }
        auto I3 = get_I3(eta);
        // 5 pi^2 / 162 prefactor from Dominik Eq. 4 / JC 1999 Eq. 28 second term.
        return forceeval(-5.0 * POW2(static_cast<double>(EIGEN_PI)) / 162.0
                          * POW2(rhoN_A3) * I3 * S * S * S);
    }

    /// Padé-resummed dipolar contribution: alpha = alpha2 / (1 - alpha3/alpha2)
    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta, const VecType& mole_fractions) const {
        auto alpha2 = get_alpha2DD(T, rho_A3, eta, mole_fractions);
        auto alpha3 = get_alpha3DD(T, rho_A3, eta, mole_fractions);
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

    MultipolarContributionJogChapman(
        const Eigen::ArrayX<double>& m,
        const Eigen::ArrayX<double>& sigma_Angstrom,
        const Eigen::ArrayX<double>& epsilon_over_k,
        const Eigen::ArrayX<double>& mustar2,
        const Eigen::ArrayX<double>& xp,
        const Eigen::ArrayX<double>& Qstar2,
        const Eigen::ArrayX<double>& nQ,
        SigmaijRule sigmaij_rule = SigmaijRule::geometric)
    // The JC dipolar branch is hard-coded geometric (it relies on the
    // collapsed sums); the sigmaij_rule argument is forwarded only to
    // the GV quadrupolar and cross-DQ branches so the user can choose
    // arithmetic-vs-geometric for the Q-side consistently with the
    // chosen polar_combining_rule. Default is geometric for parallelism
    // with the JC dipolar branch.
    : di((((xp * mustar2 > 0).cast<int>().sum() > 0)
              ? decltype(di)(DipolarContributionJogChapman(m, sigma_Angstrom, epsilon_over_k, mustar2, xp))
              : std::nullopt)),
      quad((((nQ * Qstar2 > 0).cast<int>().sum() > 0)
              ? decltype(quad)(GrossVrabec::QuadrupolarContributionGross(m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ, sigmaij_rule))
              : std::nullopt)),
      // Cross DQ requires *both* dipole and quadrupole. The GV cross-DQ class
      // expects an nmu input, which JC doesn't carry natively -- derive it
      // from xp (n_mu = m * x_p) so the GV term sees a consistent count.
      diquad((di && quad)
              ? decltype(diquad)(GrossVrabec::DipolarQuadrupolarContributionVrabecGross(
                    m, sigma_Angstrom, epsilon_over_k, mustar2,
                    (m * xp).eval(), Qstar2, nQ, sigmaij_rule))
              : std::nullopt)
    {}

    template <typename TTYPE, typename RhoType, typename EtaType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rho_A3, const EtaType& eta, const VecType& mole_fractions) const {
        using type = std::common_type_t<TTYPE, RhoType, EtaType, decltype(mole_fractions[0])>;
        type alpha2DD = 0.0, alpha3DD = 0.0, alphaDD = 0.0;
        type alpha2QQ = 0.0, alpha3QQ = 0.0, alphaQQ = 0.0;
        type alpha2DQ = 0.0, alpha3DQ = 0.0, alphaDQ = 0.0;
        if (di && di.value().has_a_polar) {
            alpha2DD = di.value().get_alpha2DD(T, rho_A3, eta, mole_fractions);
            alpha3DD = di.value().get_alpha3DD(T, rho_A3, eta, mole_fractions);
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
