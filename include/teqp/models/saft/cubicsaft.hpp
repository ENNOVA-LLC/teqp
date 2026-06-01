#pragma once

/*
 * Cubic-SAFT (a.k.a. CPCA, Cubic Plus Chain and Association)
 *
 * Reference: Sisco, Alajmi, Abutaqiya, Vargas, Chapman,
 *   "Cubic-Plus-Chain IV: A General Framework for the SAFT-Based Chain +
 *    Association Modification to the Cubic Equation of State,"
 *   Ind. Eng. Chem. Res., 2023, doi: 10.1021/acs.iecr.3c02774.
 *
 * Decomposition (Eq. 6 of Sisco 2023):
 *
 *     a^R = mbar * a_cubic + a_chain + a_assoc
 *
 * where a_cubic is the classical cubic EOS Helmholtz energy of the *monomer*
 * (RK / SRK / PR), a_chain bonds the monomers into a chain via the SAFT chain
 * contribution (Eq. 7), and a_assoc is the standard SAFT association term.
 *
 * The chain Helmholtz energy is, exactly as in PC-SAFT and CPA-with-chain:
 *
 *     a_chain = -sum_i x_i (m_i - 1) ln g^seg(eta)
 *
 * with g^seg the segment-segment radial distribution function evaluated at
 * contact. Sisco 2023 Eq. 10 uses the Elliott (1990) approximation
 * g^seg = 1 / (1 - 0.475 * eta) with eta = mbar * B / V. We accept the same
 * three RDF choices already supported by teqp's CPA:
 *   - "Elliott" (Sisco-2023 default; same as KG):  g = 1 / (1 - 1.9 * eta)
 *   - "Carnahan-Starling":                          g = (2 - eta) / (2 (1-eta)^3)
 *   - "KG"  (Kontogeorgis-Yakoumis, alias to Elliott)
 *
 * Note: Sisco's Eq. 10 with eta = mbar B / V and his "0.475" coefficient gives
 *     g = 1 / (1 - 0.475 * mbar * B / V).
 * teqp's CPA uses eta = (rho / 4) * b = (1/4) * (mbar B / V) so the same g
 * expression there is 1 / (1 - 1.9 * eta_teqp). Both forms describe the same
 * RDF; we use teqp's convention (eta_teqp = rho * b_mix / 4) for consistency
 * with the existing CPA implementation.
 *
 * Mixing rules (Sisco 2023 Eq. 12-13):
 *     mbar    = sum_i x_i m_i
 *     mbar A  = sum_{i,j} (x_i m_i)(x_j m_j) a_ij        (classical vdW1F)
 *     mbar B  = sum_i (x_i m_i) b_i                       (linear in m_i b_i)
 *     a_ij    = sqrt(a_i a_j) (1 - k_ij)
 *
 * Note that A and B above are *per-monomer* (not per-molecule) properties: the
 * cubic EOS evaluated with (A, B) and the *chain* density rho_chain = rho_mol
 * gives a^cubic per monomer, which is then multiplied by mbar to get the
 * per-molecule chain-aware contribution. This is consistent with PC-SAFT,
 * where the monomer dispersion term carries the m_bar factor at the front
 * (Gross-Sadowski 2001 Eq. A.20).
 *
 * When m_i == 1 for all species and the association term is non-trivial, this
 * model reduces to CPA. When m_i == 1 and association is off, it reduces to
 * the classical cubic EOS. When association is off (but chain is active), it
 * reduces to the CPC EOS (Sisco-Abutaqiya 2019).
 */

#include "nlohmann/json.hpp"
#include <Eigen/Dense>
#include <optional>
#include "teqp/types.hpp"
#include "teqp/constants.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/math/pow_templates.hpp"
#include "teqp/models/saft/polar_terms.hpp"
#include "teqp/models/saft/peneloux_shift.hpp"
#include "teqp/models/association/association_types.hpp"

namespace teqp::saft::cubicsaft {

template<typename X> auto POW2_local(X x) { return x * x; };
template<typename X> auto POW3_local(X x) { return x * POW2_local(x); };

/// Cubic-EOS variant selector. The two delta values are functions of this.
enum class CubicVariant { PR, SRK };

inline CubicVariant parse_cubic_variant(const std::string& s) {
    if (s == "PR" || s == "peng-robinson" || s == "PR76" || s == "PR78") return CubicVariant::PR;
    if (s == "SRK" || s == "RKS" || s == "soave-redlich-kwong") return CubicVariant::SRK;
    throw teqp::InvalidArgument("Unrecognized cubic variant: " + s);
}

/// Segment-RDF selector for the chain term. Mirrors teqp's CPA radial_dists.
enum class ChainRDF { Elliott, CarnahanStarling };

inline ChainRDF parse_chain_rdf(const std::string& s) {
    // Elliott and KG are mathematically identical; accept both spellings.
    if (s == "Elliott" || s == "elliott" || s == "KG" || s == "kg") return ChainRDF::Elliott;
    if (s == "CS" || s == "cs" || s == "Carnahan-Starling" || s == "carnahan-starling") return ChainRDF::CarnahanStarling;
    throw teqp::InvalidArgument("Unrecognized chain RDF: " + s + " (expected 'Elliott'/'KG' or 'CS')");
}

/**
 * \brief Selector for the paired (alpha, beta) temperature-dependence scheme.
 *
 * \details Conceptually analogous to choosing a Soave alpha function: the
 * scheme is an *EOS-family* property, not a per-species fittable parameter.
 * Alajmi-Sisco 2022 is a package deal that replaces *both* the user-supplied
 * Soave kappa on the attraction side *and* introduces a covolume (v0)
 * temperature dependence on the repulsive side, with all three constants
 * coming from the same MW correlation set.
 *
 *   None             -> Soave alpha with user-supplied c1 (kappa_alpha) on a(T)
 *                       and T-independent b = b_0 (the existing baseline).
 *   AlajmiSisco2022  -> kappa_alpha, kappa_beta1, kappa_beta2 all replaced by
 *                       Alajmi-Sisco 2022 Table 1 MW correlations:
 *                         a(T) = a_0 * (1 + kappa_alpha * (1 - sqrt(T/Tc)))^2
 *                         b(T) = b_0 * kappa_beta1 * exp(-kappa_beta2 * T/Tc)
 */
enum class AlphaBetaScheme { None, AlajmiSisco2022 };

inline AlphaBetaScheme parse_alpha_beta_scheme(const std::string& s) {
    if (s.empty() || s == "none" || s == "None") return AlphaBetaScheme::None;
    if (s == "alajmi-2022" || s == "alajmi-sisco-2022" || s == "AlajmiSisco2022") {
        return AlphaBetaScheme::AlajmiSisco2022;
    }
    throw teqp::InvalidArgument(
        "Unrecognized alpha_beta_scheme: " + s + " (expected 'none' or 'alajmi-2022')");
}

/// Alajmi-Sisco 2022 constants for a single species at MW [g/mol].
struct AlajmiConstants {
    double kappa_alpha;   ///< replaces user-supplied c1 in the Soave alpha function
    double kappa_beta1;   ///< b(T) prefactor in beta_T(T) = kappa_beta1 * exp(-kappa_beta2 * T/Tc)
    double kappa_beta2;   ///< b(T) exponential decay rate (in reduced T)
};

/**
 * \brief Alajmi-Sisco 2022 Table 1 MW correlations.
 *
 * \details Returns the trio (kappa_alpha, kappa_beta1, kappa_beta2) for a single
 * species at molecular weight ``MW`` [g/mol] via the published SRK
 * correlations:
 * \f[
 *     \kappa_\alpha   = 0.2425 \cdot \text{MW}^{0.2829}
 * \f]
 * \f[
 *     \kappa_{\beta 1} = 0.7112 \cdot \text{MW}^{0.1502}
 * \f]
 * \f[
 *     \kappa_{\beta 2} = 0.1549 \cdot \ln(\text{MW}) - 0.3224
 * \f]
 *
 * The Alajmi-Sisco 2022 paper fits these constants against SRK reference data.
 * No PR-specific refit has been published, so calling this with
 * ``CubicVariant::PR`` raises ``InvalidArgument``. Remove the guard once a
 * validated PR coefficient set is available.
 */
inline AlajmiConstants alajmi_kappa(double MW, CubicVariant variant) {
    if (MW <= 0) {
        throw teqp::InvalidArgument(
            "alajmi_kappa requires MW > 0; got " + std::to_string(MW));
    }
    if (variant == CubicVariant::PR) {
        throw teqp::InvalidArgument(
            "alpha_beta_scheme='alajmi-2022' is only validated for SRK in "
            "Alajmi-Sisco 2022. PR-monomer constants have not been published; "
            "select 'none' for PR-cubic-SAFT or use SRK-cubic-SAFT.");
    }
    return AlajmiConstants{
        /*kappa_alpha=*/0.2425 * std::pow(MW, 0.2829),
        /*kappa_beta1=*/0.7112 * std::pow(MW, 0.1502),
        /*kappa_beta2=*/0.1549 * std::log(MW) - 0.3224,
    };
}

/// Per-fluid coefficients for the cubic-SAFT family.
struct CubicSAFTCoeffs {
    std::string name;
    double m = 1.0;                  ///< segment number (mi); m=1 reduces to CPA
    double a0i = -1;                 ///< Pa * m^6 / mol^2; monomer attraction at Tr=1
    double bi  = -1;                 ///< m^3 / mol; monomer covolume (b_0, T-independent input)
    double c1  = -1;                 ///< Soave alpha-function prefactor (kappa_alpha); ignored under alpha_beta_scheme != None
    double Tc  = -1;                 ///< K; critical temperature
    double MW  = -1;                 ///< g/mol; required when alpha_beta_scheme != None
    double volume_shift = 0.0;       ///< Péneloux volume shift, T-independent [m^3/mol]
    std::string BibTeXKey;
    // Polar parameters (reuse SAFT conventions; zero -> not active). When any
    // polar parameter is set, epsilon_over_k must be supplied — there is no
    // sensible default for the polar T* reduction in a cubic framework.
    double epsilon_over_k = 0;       ///< K; required when polar/quadrupolar active
    double mustar2 = 0;
    double nmu = 0;
    double xp = 0;                   ///< Jog-Chapman polar-segment fraction
    double Qstar2 = 0;
    double nQ = 0;
};

/**
 * Cubic monomer term: standard PR/SRK on the *monomer* with classical vdW1F
 * mixing on (a, b). Returns a^cubic *per monomer*. The CubicSAFT umbrella
 * multiplies this by mbar to get the per-molecule chain-aware contribution.
 */
class CubicMonomer {
private:
    const CubicVariant variant;
    const Eigen::ArrayXd a0i, b0i, c1, Tc;
    const Eigen::ArrayXd kappa_beta1, kappa_beta2;
    const std::optional<Eigen::ArrayXXd> kmat;
    double delta_1, delta_2;
    const double R_gas;
public:
    /// ``b0i`` is the T-independent covolume input (Alajmi-Sisco 2022's "b_0").
    /// ``c1`` is the Soave alpha-function prefactor (kappa_alpha): either
    /// user-supplied (default scheme) or filled in by the umbrella from a
    /// published correlation (e.g. Alajmi-Sisco 2022's kappa_alpha). 
    /// ``kappa_beta1``, ``kappa_beta2`` are the covolume T-dependence parameters:
    ///     beta_T(T) = kappa_beta1 * exp(-kappa_beta2 * T_r),    T_r = T / T_C
    ///     b_i(T)    = b0_i * beta_T_i(T)
    /// Setting kappa_beta1=1, kappa_beta2=0 gives beta_T=1 identically (the
    /// T-independent baseline). The umbrella class (CubicSAFTNonpolarMixture)
    /// is responsible for filling these from the selected alpha_beta_scheme.
    CubicMonomer(CubicVariant variant,
                 const Eigen::ArrayXd& a0i,
                 const Eigen::ArrayXd& b0i,
                 const Eigen::ArrayXd& c1,
                 const Eigen::ArrayXd& Tc,
                 const Eigen::ArrayXd& kappa_beta1,
                 const Eigen::ArrayXd& kappa_beta2,
                 double R_gas,
                 std::optional<Eigen::ArrayXXd> kmat = std::nullopt)
      : variant(variant), a0i(a0i), b0i(b0i), c1(c1), Tc(Tc),
        kappa_beta1(kappa_beta1), kappa_beta2(kappa_beta2),
        kmat(kmat), R_gas(R_gas)
    {
        if (variant == CubicVariant::PR) {
            delta_1 = 1.0 + std::sqrt(2.0);
            delta_2 = 1.0 - std::sqrt(2.0);
        } else {
            delta_1 = 1.0;
            delta_2 = 0.0;
        }
        const auto N = a0i.size();
        if (b0i.size() != N || c1.size() != N || Tc.size() != N
            || kappa_beta1.size() != N || kappa_beta2.size() != N) {
            throw teqp::InvalidArgument("CubicMonomer: a0i / b0i / c1 / Tc / kappa_beta1 / kappa_beta2 must all be the same length");
        }
    }

    std::size_t size() const { return static_cast<std::size_t>(a0i.size()); }

    template<typename TType>
    auto get_ai(const TType& T, Eigen::Index i) const {
        // Soave alpha function: a(T) = a0 * (1 + c1 * (1 - sqrt(T/Tc)))^2
        return forceeval(a0i[i] * POW2_local(1.0 + c1[i] * (1.0 - sqrt(T / Tc[i]))));
    }

    /// Alajmi-Sisco 2022 Eq. 11 T-dependent covolume:
    ///     b_i(T) = b0_i * kappa_beta1_i * exp(-kappa_beta2_i * T / Tc_i)
    /// When kappa_beta1 = 1 and kappa_beta2 = 0 (defaults), returns b0_i.
    template<typename TType>
    auto get_bi(const TType& T, Eigen::Index i) const {
        return forceeval(b0i[i] * kappa_beta1[i] * exp(-kappa_beta2[i] * T / Tc[i]));
    }

    /// Per-species b(T) for all species (used by the chain / polar / association
    /// layers that need the T-dependent covolume vector).
    template<typename TType>
    auto get_b_vector(const TType& T) const {
        using out_t = std::decay_t<TType>;
        Eigen::ArrayX<out_t> b(a0i.size());
        for (Eigen::Index i = 0; i < a0i.size(); ++i) {
            b[i] = get_bi(T, i);
        }
        return b;
    }

    /// Returns (A, B) — the *monomer* attractive and covolume coefficients
    /// after vdW1F mixing on the segment basis. Caller (CubicSAFT) is
    /// responsible for the mbar factor on the segment-basis composition.
    /// b_i is the Alajmi-Sisco T-dependent covolume (reduces to b0_i when
    /// kappa_beta defaults are used).
    template<typename TType, typename VecType>
    auto get_AB_segment(const TType& T, const VecType& segment_fractions) const {
        using return_type = std::common_type_t<TType, decltype(segment_fractions[0])>;
        return_type A = 0.0, B = 0.0;
        for (Eigen::Index i = 0; i < a0i.size(); ++i) {
            B += segment_fractions[i] * get_bi(T, i);
            auto ai = get_ai(T, i);
            for (Eigen::Index j = 0; j < a0i.size(); ++j) {
                auto aj = get_ai(T, j);
                double kij = kmat ? kmat.value()(i, j) : 0.0;
                A += segment_fractions[i] * segment_fractions[j] * (1.0 - kij) * sqrt(ai * aj);
            }
        }
        return std::make_tuple(A, B);
    }

    /// Sisco 2023 Eq. 11 cubic Helmholtz evaluated at the chain reduced
    /// volume beta = mbar * B / V = B * rho_seg = B * mbar * rho_molar.
    ///
    /// Inputs are *segment-basis*: segment_fractions x_seg,i = x_i m_i / mbar
    /// and rho_seg = mbar * rho_molar. The returned value is what Sisco calls
    /// a_cubic (Eq. 11) — i.e., the cubic contribution that the chain umbrella
    /// will multiply by mbar to get the per-molecule contribution (Eq. 6).
    ///
    /// Sisco's mixing rule (Eq. 12) gives A = mbar * A_seg where
    ///     A_seg = sum_ij x_seg,i x_seg,j a_ij.
    /// So we compute A_seg here and multiply by mbar at the call site. The
    /// caller passes mbar explicitly so this class stays self-contained.
    template<typename TType, typename RhoType, typename VecType, typename MType>
    auto alphar_Sisco_Eq11(const TType& T, const RhoType& rho_seg,
                            const VecType& segment_fractions, const MType& mbar) const {
        auto [A_seg, B] = get_AB_segment(T, segment_fractions);
        auto A = mbar * A_seg;  // Sisco Eq. 12: A = mbar * A_seg
        // Repulsive: -ln(1 - B * rho_seg)
        // Attractive: -A / (B R T (d1-d2)) * ln((1 + d1 * B * rho_seg)/(1 + d2 * B * rho_seg))
        return forceeval(
            -log(1.0 - B * rho_seg)
            - A / (R_gas * T * B * (delta_1 - delta_2))
              * log((delta_1 * B * rho_seg + 1.0) / (delta_2 * B * rho_seg + 1.0))
        );
    }
};

/**
 * SAFT chain term applied to a cubic monomer. Sisco 2023 Eq. 7.
 *
 *     a_chain / NkT = -sum_i x_i (m_i - 1) ln g^seg(eta_chain)
 *
 * with eta_chain = rho_molar * b_mix(T) / 4 in teqp's CPA convention,
 * where b_mix(T) = sum_i x_i m_i b_i(T) is the segment-weighted T-dependent
 * covolume of the mixture and rho_molar is the *molecular* density.
 *
 * b_i(T) follows the covolume T-dependence of the active alpha_beta scheme:
 *     b_i(T) = b0_i * kappa_beta1_i * exp(-kappa_beta2_i * T/Tc_i)
 * Setting kappa_beta1=1, kappa_beta2=0 recovers the T-independent b0_i (the
 * "None" scheme). The umbrella (CubicSAFTNonpolarMixture) is responsible for
 * filling these arrays from the selected alpha_beta_scheme.
 */
class CubicChain {
private:
    const Eigen::ArrayXd m_segments, b0i, Tc, kappa_beta1, kappa_beta2;
    const ChainRDF rdf;

    /// T-dependent covolume per species [m^3/mol]. Alajmi-Sisco 2022 Eq. 11.
    template<typename TType>
    auto bi_of_T(const TType& T, Eigen::Index i) const {
        return forceeval(b0i[i] * kappa_beta1[i] * exp(-kappa_beta2[i] * T / Tc[i]));
    }

public:
    CubicChain(const Eigen::ArrayXd& m_segments,
               const Eigen::ArrayXd& b0i,
               const Eigen::ArrayXd& Tc,
               const Eigen::ArrayXd& kappa_beta1,
               const Eigen::ArrayXd& kappa_beta2,
               ChainRDF rdf)
      : m_segments(m_segments), b0i(b0i), Tc(Tc),
        kappa_beta1(kappa_beta1), kappa_beta2(kappa_beta2),
        rdf(rdf)
    {
        const auto N = m_segments.size();
        if (b0i.size() != N || Tc.size() != N
            || kappa_beta1.size() != N || kappa_beta2.size() != N) {
            throw teqp::InvalidArgument("CubicChain: m / b0i / Tc / kappa_beta1 / kappa_beta2 must all be the same length");
        }
    }

    /// Mixture covolume b_mix(T) [m^3/mol]: sum_i x_i m_i b_i(T).
    template<typename TType, typename VecType>
    auto bmix(const TType& T, const VecType& mole_fractions) const {
        using sum_t = std::common_type_t<TType, decltype(mole_fractions[0])>;
        sum_t b = 0.0;
        for (Eigen::Index i = 0; i < m_segments.size(); ++i) {
            b += mole_fractions[i] * m_segments[i] * bi_of_T(T, i);
        }
        return forceeval(b);
    }

    /// Contribution to alpha^r per molecule. rho_molar is mol/m^3.
    template<typename TType, typename RhoType, typename VecType>
    auto alphar(const TType& T, const RhoType& rho_molar, const VecType& mole_fractions) const {
        auto b_mix = bmix(T, mole_fractions);
        // teqp CPA convention: eta = rho * b / 4
        auto eta = forceeval((rho_molar / 4.0) * b_mix);

        // Segment-segment RDF at contact
        using g_t = std::common_type_t<decltype(eta), double>;
        g_t g_seg;
        switch (rdf) {
            case ChainRDF::Elliott:
                // Elliott / Kontogeorgis-Yakoumis form: g = 1/(1 - 1.9 eta)
                g_seg = 1.0 / (1.0 - 1.9 * eta);
                break;
            case ChainRDF::CarnahanStarling:
                // Carnahan-Starling: g = (2 - eta) / (2 (1 - eta)^3)
                g_seg = (2.0 - eta) / (2.0 * POW3_local(1.0 - eta));
                break;
        }

        // a_chain = -sum_i x_i (m_i - 1) ln g_seg
        // g_seg depends on x via b_mix(x), but the same g_seg appears at every
        // i, so it pulls out of the i-sum. (It does *not* drop out of the
        // composition derivative, which is one of the reasons the chain
        // term is more complex than it looks.)
        using sum_t = std::common_type_t<decltype(mole_fractions[0]), decltype(g_seg)>;
        sum_t mm1_avg = 0.0;
        for (Eigen::Index i = 0; i < m_segments.size(); ++i) {
            mm1_avg += mole_fractions[i] * (m_segments[i] - 1.0);
        }
        return forceeval(-mm1_avg * log(g_seg));
    }

    /// Expose b_mix for the umbrella density solver. Public, double-precision
    /// version used by GenericSAFT::get_bmix. Uses the supplied T to evaluate
    /// the Alajmi-Sisco T-dependent covolume.
    template<typename VecType>
    double get_bmix(double T, const VecType& mole_fractions) const {
        double b = 0.0;
        for (Eigen::Index i = 0; i < m_segments.size(); ++i) {
            b += mole_fractions[i] * m_segments[i]
                 * (b0i[i] * kappa_beta1[i] * std::exp(-kappa_beta2[i] * T / Tc[i]));
        }
        return b;
    }

    /// Per-species T-dependent b vector (used by the umbrella to derive d(T)
    /// for the polar layer via the AM 2024 d = (3 b / (2 pi N_A))^(1/3) formula).
    template<typename TType>
    auto get_b_vector(const TType& T) const {
        using out_t = std::decay_t<TType>;
        Eigen::ArrayX<out_t> b(m_segments.size());
        for (Eigen::Index i = 0; i < m_segments.size(); ++i) {
            b[i] = bi_of_T(T, i);
        }
        return b;
    }
};

/**
 * \brief Cubic-SAFT nonpolar mixture: cubic monomer + SAFT chain + optional
 *        polar / quadrupolar / Alajmi-Sisco T-dependent covolume.
 *
 * \details Slots into ``GenericSAFT`` as a ``NonPolarTerms`` variant sibling
 * to ``PCSAFTMixture``. Implements the Sisco 2023 CPCA framework (cubic
 * monomer + Wertheim chain) with optional Jog-Chapman polar following the
 * Abutaqiya-Marshall 2024 SP-SRK conventions.
 *
 * \par Polar conventions (AM 2024 SP-SRK)
 * Only the JC polar kernel is supported for cubic-SAFT. GV requires a SAFT-
 * style per-segment \f$\epsilon/k_B\f$ for its \f$T^*_{ij}\f$ argument that
 * cubic monomers do not natively carry; the umbrella layer (e.g. fpFlash's
 * adapter) is expected to reject ``polar_model = GrossVrabec`` upstream.
 *
 * The polar pair-distance is derived from the cubic covolume via AM 2024
 * Eq. 14:
 * \f[
 *    d_i = (3 b_i / (2 \pi N_A))^{1/3}
 * \f]
 * (note: \f$4^{1/3} \approx 1.587\f$ smaller than the SAFT segment-volume
 * relation \f$\sigma = (6 b / (\pi N_A))^{1/3}\f$; the AM 2024 form is the
 * physically correct hard-sphere diameter for a vdW-style covolume).
 *
 * The polar branch defaults to a geometric \f$\sigma_{ij}\f$ combining rule.
 * AM 2024 Eq. 15 notes this is technically inconsistent with the arithmetic
 * mixing used in the cubic dispersion term, but the accuracy cost is
 * \f$\sim\f$2 % for typical HC mixtures with \f$\sim\f$3 SD of size variation
 * and the O(N) collapse of JC's pair sums is the formulation's main appeal.
 * JC's collapsed kernels are only valid under the geometric assumption; the
 * arithmetic path falls back to full O(N\f$^2\f$) / O(N\f$^3\f$) sums.
 *
 * \par Alpha-beta scheme (optional T-dependence on both a and b)
 * Selected at the mixture level via ``AlphaBetaScheme``. ``None`` (default)
 * keeps Soave's alpha function with the user-supplied \f$\kappa_\alpha\f$
 * (CubicSAFTCoeffs::c1) and a T-independent \f$b = b_0\f$.
 *
 * ``AlajmiSisco2022`` is a *package deal*: the three constants
 * \f$(\kappa_\alpha, \kappa_{\beta 1}, \kappa_{\beta 2})\f$ are all replaced by
 * Alajmi-Sisco 2022 Table 1 MW correlations (see ::alajmi_kappa). Both the
 * attraction and the covolume become temperature-dependent:
 * \f[
 *     a_i(T) = a_{0,i} \cdot \left[ 1 + \kappa_\alpha (1 - \sqrt{T/T_c}) \right]^2
 * \f]
 * \f[
 *     b_i(T) = b_{0,i} \cdot \kappa_{\beta 1} \exp(-\kappa_{\beta 2} T / T_{c,i})
 * \f]
 * The user-supplied ``c1`` is ignored when this scheme is active. The polar
 * pair-distance \f$d_i(T)\f$ is recomputed per call from \f$b_i(T)\f$. The
 * association layer (when present) continues to use \f$b_{0,i}\f$. Requires ``MW`` to be set on each coeff.
 * Currently SRK-only; PR-monomer raises until a PR-specific refit is published.
 *
 * \par Péneloux volume shift (optional)
 * Per-species ``volume_shift`` \f$c_i\f$ (default 0) enables the textbook
 * Péneloux (1982) translation. ``alphar`` evaluates all sub-terms (cubic,
 * chain, polar) at the shifted density
 * \f$\tilde\rho = \rho / (1 + \rho \bar c)\f$ and adds the closing
 * \f$-\ln(1 + \rho \bar c)\f$ at the user's \f$\rho\f$. \f$c_i = 0\f$ for
 * all species recovers the unshifted EOS bit-exactly. See
 * peneloux_shift.hpp for the algebra and K-value invariance.
 */
class CubicSAFTNonpolarMixture {
public:
    using DipolarGV   = teqp::saft::polar_terms::GrossVrabec::DipolarContributionGrossVrabec;
    using DipolarJC   = teqp::saft::polar_terms::JogChapman::DipolarContributionJogChapman;
    using QuadGV      = teqp::saft::polar_terms::GrossVrabec::QuadrupolarContributionGross;

protected:
    Eigen::ArrayXd m_segments, bi, Tc, kappa_beta1, kappa_beta2,
                   sigma_Angstrom, epsilon_over_k,
                   volume_shift;   ///< Péneloux T-independent volume shift per species [m^3/mol] (zero by default)
    std::vector<std::string> names, bibtex;
    CubicMonomer monomer;
    CubicChain   chain;
    std::optional<DipolarGV> dipolar;
    std::optional<DipolarJC> dipolar_jc;
    std::optional<QuadGV>    quadrupolar;
    const double R_gas;

    /// Polar pair-distance scale used by the dipolar / quadrupolar terms.
    ///
    /// Abutaqiya & Marshall (2024), AIChE J. 70, e18451, Eq. 14:
    ///     d_i = (3 b_i / (2 pi N_A))^{1/3}  [in m, when b_i is in m^3/mol]
    /// derived from b = 4 V_molecular = (2 pi / 3) d^3. This is *different*
    /// from the SAFT segment-volume relation sigma = (6 b / (pi N_A))^{1/3}
    /// by a factor of 4^{1/3} ~ 1.587; the AM 2024 d is the physically
    /// correct hard-sphere diameter for a vdW-style covolume.
    ///
    /// In Angstrom with b_i in m^3/mol: d[A] = (3 b / (2 pi N_A))^{1/3} * 1e10
    static Eigen::ArrayXd sigma_from_b(const Eigen::ArrayXd& bi) {
        Eigen::ArrayXd sig(bi.size());
        for (Eigen::Index i = 0; i < bi.size(); ++i) {
            sig[i] = std::cbrt(3.0 * bi[i] / (2.0 * EIGEN_PI * N_A)) * 1e10;
        }
        return sig;
    }

private:
    /// Helper to validate and unpack a per-field array from coeffs.
    static Eigen::ArrayXd unpack(const std::vector<CubicSAFTCoeffs>& cs, double CubicSAFTCoeffs::*field) {
        Eigen::ArrayXd out(cs.size());
        for (Eigen::Index i = 0; i < (Eigen::Index)cs.size(); ++i) {
            out[i] = cs[i].*field;
        }
        return out;
    }

    static void validate_coeffs(const std::vector<CubicSAFTCoeffs>& cs,
                                 AlphaBetaScheme scheme) {
        const bool need_MW = (scheme != AlphaBetaScheme::None);
        for (const auto& c : cs) {
            if (c.a0i <= 0 || c.bi <= 0) {
                throw teqp::InvalidArgument(
                    "CubicSAFT: a0i and bi must be positive for fluid " + c.name);
            }
            if (c.m <= 0) {
                throw teqp::InvalidArgument(
                    "CubicSAFT: m must be positive for fluid " + c.name);
            }
            if (need_MW && c.MW <= 0) {
                throw teqp::InvalidArgument(
                    "CubicSAFT: MW must be positive when alpha_beta_scheme != 'none' "
                    "for fluid " + c.name);
            }
            bool polar_active = c.mustar2 != 0 || c.Qstar2 != 0;
            if (polar_active && c.epsilon_over_k <= 0) {
                throw teqp::InvalidArgument(
                    "CubicSAFT: epsilon_over_k must be supplied (positive) when "
                    "polar or quadrupolar parameters are set for fluid " + c.name);
            }
            if (c.volume_shift >= c.bi) {
                throw teqp::InvalidArgument(
                    "CubicSAFT: volume_shift (" + std::to_string(c.volume_shift)
                    + " m^3/mol) must be strictly less than bi ("
                    + std::to_string(c.bi) + " m^3/mol) for fluid " + c.name
                    + ". The Peneloux-translated covolume b_t = bi - "
                    "volume_shift would be non-positive, indicating the "
                    "cubic critical properties (Tc, Pc) give a covolume "
                    "incompatible with the target density. Recalibrate Tc/Pc "
                    "for this species or accept zero shift.");
            }
        }
    }

    /// Resolved per-species (c1, kappa_beta1, kappa_beta2) arrays for the
    /// active scheme. ``None`` keeps the user-supplied c1 with (1, 0)
    /// covolume; Alajmi overwrites all three from MW correlations.
    struct ResolvedAlphaBeta {
        Eigen::ArrayXd c1, kappa_beta1, kappa_beta2;
    };

    /// Validate inputs and resolve (c1, kappa_beta1, kappa_beta2) from the
    /// active scheme in a single pass. Throws InvalidArgument on bad input.
    static ResolvedAlphaBeta validate_and_resolve(const std::vector<CubicSAFTCoeffs>& cs,
                                                    AlphaBetaScheme scheme,
                                                    CubicVariant variant) {
        validate_coeffs(cs, scheme);
        const Eigen::Index N = static_cast<Eigen::Index>(cs.size());
        ResolvedAlphaBeta out{Eigen::ArrayXd(N), Eigen::ArrayXd(N), Eigen::ArrayXd(N)};
        if (scheme == AlphaBetaScheme::None) {
            for (Eigen::Index i = 0; i < N; ++i) {
                out.c1[i]          = cs[i].c1;
                out.kappa_beta1[i] = 1.0;
                out.kappa_beta2[i] = 0.0;
            }
        } else { // AlajmiSisco2022 — package deal: c1 is overwritten too.
            for (Eigen::Index i = 0; i < N; ++i) {
                auto k = alajmi_kappa(cs[i].MW, variant);
                out.c1[i]          = k.kappa_alpha;
                out.kappa_beta1[i] = k.kappa_beta1;
                out.kappa_beta2[i] = k.kappa_beta2;
            }
        }
        return out;
    }

public:
    CubicSAFTNonpolarMixture(const std::vector<CubicSAFTCoeffs>& coeffs_in,
                              CubicVariant variant,
                              ChainRDF rdf,
                              AlphaBetaScheme scheme,
                              double R_gas,
                              const std::optional<Eigen::ArrayXXd>& kmat = std::nullopt)
      : CubicSAFTNonpolarMixture(coeffs_in, variant, rdf, R_gas, kmat,
                                  validate_and_resolve(coeffs_in, scheme, variant))
    {}

private:
    /// Delegated constructor that consumes the resolved (c1, kappa_beta)
    /// arrays. validate_and_resolve runs before any member-init via the
    /// public constructor's delegate-target call.
    CubicSAFTNonpolarMixture(const std::vector<CubicSAFTCoeffs>& coeffs_in,
                              CubicVariant variant,
                              ChainRDF rdf,
                              double R_gas,
                              const std::optional<Eigen::ArrayXXd>& kmat,
                              ResolvedAlphaBeta ab)
      : m_segments(unpack(coeffs_in, &CubicSAFTCoeffs::m)),
        bi(unpack(coeffs_in, &CubicSAFTCoeffs::bi)),  // b0i (T-independent input)
        Tc(unpack(coeffs_in, &CubicSAFTCoeffs::Tc)),
        kappa_beta1(ab.kappa_beta1),
        kappa_beta2(ab.kappa_beta2),
        sigma_Angstrom(sigma_from_b(unpack(coeffs_in, &CubicSAFTCoeffs::bi))),  // T-independent reference (used as fallback when polar diameter doesn't need T)
        epsilon_over_k(unpack(coeffs_in, &CubicSAFTCoeffs::epsilon_over_k)),
        volume_shift(unpack(coeffs_in, &CubicSAFTCoeffs::volume_shift)),
        monomer(CubicMonomer(
            variant,
            unpack(coeffs_in, &CubicSAFTCoeffs::a0i),
            unpack(coeffs_in, &CubicSAFTCoeffs::bi),
            ab.c1,
            unpack(coeffs_in, &CubicSAFTCoeffs::Tc),
            ab.kappa_beta1,
            ab.kappa_beta2,
            R_gas, kmat)),
        chain(CubicChain(
            unpack(coeffs_in, &CubicSAFTCoeffs::m),
            unpack(coeffs_in, &CubicSAFTCoeffs::bi),
            unpack(coeffs_in, &CubicSAFTCoeffs::Tc),
            ab.kappa_beta1,
            ab.kappa_beta2,
            rdf)),
        R_gas(R_gas)
    {
        const auto& coeffs = coeffs_in;  // alias for clarity in body
        names.reserve(coeffs.size());
        bibtex.reserve(coeffs.size());
        for (const auto& c : coeffs) {
            names.push_back(c.name);
            bibtex.push_back(c.BibTeXKey);
        }

        // Build polar contributions when active. JC takes precedence when xp
        // is set on any species (Marshall convention); otherwise GV with nmu.
        auto mustar2 = unpack(coeffs, &CubicSAFTCoeffs::mustar2);
        auto nmu     = unpack(coeffs, &CubicSAFTCoeffs::nmu);
        auto xp      = unpack(coeffs, &CubicSAFTCoeffs::xp);
        auto Qstar2  = unpack(coeffs, &CubicSAFTCoeffs::Qstar2);
        auto nQ      = unpack(coeffs, &CubicSAFTCoeffs::nQ);

        // See class docstring for the AM 2024 SP-SRK conventions
        // (geometric sigma_ij, JC-only polar, AM 2024 d formula).
        const auto polar_sigmaij = teqp::saft::polar_terms::SigmaijRule::geometric;

        if ((mustar2.abs() * xp.abs()).sum() > 0) {
            // Convert the shared GV-convention (mu*)^2 input to the
            // per-segment mu_JC^2 in SI that the JC kernel consumes. See
            // PCSAFTMixture::build_dipolar_jc docstring for the formula.
            constexpr double FOUR_PI_EPS0 = 1.11265005605362e-10;
            constexpr double K_B = 1.380649e-23;
            Eigen::ArrayXd mu_squared_SI(mustar2.size());
            for (Eigen::Index k = 0; k < mustar2.size(); ++k) {
                const double sigma_m = sigma_Angstrom[k] * 1e-10;
                const double sigma3 = sigma_m * sigma_m * sigma_m;
                mu_squared_SI[k] = mustar2[k] * FOUR_PI_EPS0
                                    * epsilon_over_k[k] * K_B * sigma3;
            }
            dipolar_jc.emplace(DipolarJC(m_segments, xp, mu_squared_SI));
        } else if ((mustar2.abs() * nmu.abs()).sum() > 0) {
            dipolar.emplace(DipolarGV(
                m_segments, sigma_Angstrom, epsilon_over_k, mustar2, nmu,
                polar_sigmaij));
        }
        if ((Qstar2.abs() * nQ.abs()).sum() > 0) {
            quadrupolar.emplace(QuadGV(
                m_segments, sigma_Angstrom, epsilon_over_k, Qstar2, nQ,
                polar_sigmaij));
        }
    }

public:
    template<class VecType>
    auto R(const VecType& /*molefrac*/) const { return R_gas; }

    /**
     * \brief Translated mixture covolume seen by the density solver.
     *
     * \details Returns the Péneloux-translated covolume
     * \f$b_{t,\text{mix}} = b_{o,\text{mix}} - \bar c\f$, with the two
     * terms following *different* mixing rules:
     *   - \f$b_{o,\text{mix}} = \sum_i x_i m_i b_{o,i}\f$:
     *     Sisco 2023 CPC chain-mixing (a per-segment physics derivation).
     *   - \f$\bar c = \sum_i x_i c_i\f$:
     *     Palma 2018 Eq. 3 linear-in-x (a per-molecule empirical
     *     correction, matching the PC-SAFT convention).
     *
     * Identical to the unshifted case for pure species and for any \f$m=1\f$
     * mixture (the CPA limit). For \f$m>1\f$ mixtures the rules diverge.
     *
     * This is what scales the density solver's iteration variable
     * \f$\eta_\text{solver} = b_{t,\text{mix}} \rho_\text{shifted}\f$,
     * so the cap \f$\eta < 1\f$ maps to the physical hard-sphere wall of
     * the *translated* EOS (Jaubert 2016: the translated covolume \f$b_t\f$
     * is the "true" covolume of the translated EOS; \f$b_o\f$ has no
     * special role in the shifted state). See peneloux_shift.hpp for the
     * \f$\alpha^r\f$ algebra and the cancellation that yields
     * \f$-\ln(1 - b_t \rho_s)\f$ in the translated frame.
     *
     * Internal callers (the polar \f$\eta = \tilde\rho \, b / 4\f$ in
     * ``alphar``, and the AM 2024 diameter
     * \f$d_i = (3 b_i / (2 \pi N_A))^{1/3}\f$) keep using the
     * untranslated \f$b_{o,i}\f$ via the per-species ``bi`` field.
     */
    template<typename VecType>
    double get_bmix(double /*T*/, const VecType& mole_fractions) const {
        double b_o_mix = 0.0;
        double cbar    = 0.0;
        for (Eigen::Index i = 0; i < m_segments.size(); ++i) {
            b_o_mix += mole_fractions[i] * m_segments[i] * bi[i];
            cbar    += mole_fractions[i] * volume_shift[i];
        }
        return b_o_mix - cbar;
    }

    /// Coarse model-family tag. Cubic-SAFT identifies as SAFT-family because
    /// the chain and association physics are SAFT-derived.
    teqp::cppinterface::ModelKind get_model_kind() const {
        return teqp::cppinterface::ModelKind::SAFT;
    }

    auto get_m() const { return m_segments; }
    auto get_bi() const { return bi; }
    auto get_sigma_Angstrom() const { return sigma_Angstrom; }
    auto get_epsilon_over_k_K() const { return epsilon_over_k; }
    auto get_names() const { return names; }
    auto get_BibTeXKeys() const { return bibtex; }

    /// \brief Residual molar Helmholtz energy (Sisco 2023 Eq. 6, with
    /// optional Péneloux translation).
    ///
    /// \f$\alpha^r = \bar m \, a^\text{cubic}_\text{monomer}
    /// + a_\text{chain} + a_\text{polar} - \ln(1 + \rho \bar c)\f$,
    /// with all sub-terms evaluated at the shifted density
    /// \f$\tilde\rho = \rho / (1 + \rho \bar c)\f$.
    template<typename TType, typename RhoType, typename VecType>
    auto alphar(const TType& T, const RhoType& rho_molar, const VecType& mole_fractions) const {
        auto rho_tilde = teqp::peneloux::shifted_density(
            rho_molar, mole_fractions, volume_shift);

        // mbar = sum_i x_i m_i; segment-basis composition x_seg,i = x_i m_i / mbar
        // feeds Sisco's per-segment cubic mixing rules.
        using sum_t = std::common_type_t<TType, RhoType, decltype(mole_fractions[0])>;
        sum_t mbar = 0.0;
        for (Eigen::Index i = 0; i < m_segments.size(); ++i) {
            mbar += mole_fractions[i] * m_segments[i];
        }
        Eigen::Array<sum_t, Eigen::Dynamic, 1> x_seg(m_segments.size());
        for (Eigen::Index i = 0; i < m_segments.size(); ++i) {
            x_seg[i] = (mole_fractions[i] * m_segments[i]) / mbar;
        }
        auto rho_seg = forceeval(mbar * rho_tilde);

        // 1. Cubic (Sisco Eq. 11) scaled by mbar (Sisco Eq. 6).
        auto a_cubic = monomer.alphar_Sisco_Eq11(T, rho_seg, x_seg, mbar);
        auto alphar = forceeval(mbar * a_cubic);

        // 2. SAFT chain.
        alphar += chain.alphar(T, rho_tilde, mole_fractions);

        // 3. Optional polar / quadrupolar; eta and rho_A3 share the chain's
        // packing fraction (all derived from rho_tilde).
        if (dipolar || dipolar_jc || quadrupolar) {
            auto b_mix = chain.bmix(T, mole_fractions);
            auto eta = forceeval((rho_tilde / 4.0) * b_mix);
            auto rho_A3 = forceeval(rho_tilde * N_A * 1e-30);

            if (dipolar) {
                alphar += dipolar.value().eval(T, rho_A3, eta, mole_fractions).alpha;
            }
            if (dipolar_jc) {
                // JC requires per-call diameter; AM 2024 Eq. 14:
                // d_i [A] = (3 b_i(T) / (2 pi N_A))^(1/3) * 1e10.
                auto b_vec_T = chain.get_b_vector(T);
                using d_t = std::decay_t<decltype(b_vec_T[0])>;
                Eigen::ArrayX<d_t> d_T(b_vec_T.size());
                const double prefactor_third = 3.0 / (2.0 * EIGEN_PI * N_A);
                for (Eigen::Index i = 0; i < b_vec_T.size(); ++i) {
                    d_T[i] = pow(prefactor_third * b_vec_T[i], 1.0/3.0) * 1e10;
                }
                alphar += dipolar_jc.value().eval(T, rho_A3, eta, mole_fractions, d_T).alpha;
            }
            if (quadrupolar) {
                alphar += quadrupolar.value().eval(T, rho_A3, eta, mole_fractions).alpha;
            }
        }

        // 4. Péneloux closing correction (zero by default).
        alphar += teqp::peneloux::log_correction(
            rho_molar, mole_fractions, volume_shift);

        return forceeval(alphar);
    }
};

/// JSON-based factory for CubicSAFT.
///
/// Expected JSON shape:
///   {
///     "cubic": "PR" | "SRK",
///     "radial_dist": "Elliott" | "CS",          // optional, default "Elliott"
///     "alpha_beta_scheme": "none" | "alajmi-2022",  // optional, default "none"
///     "R_gas / J/mol/K": 8.314...,
///     "kmat": [[...]],                          // optional, NxN
///     "coeffs": [
///       {
///         "name": "...",
///         "m": 1.0,
///         "a0i / Pa m^6/mol^2": ...,
///         "bi / m^3/mol": ...,
///         "c1": ...,                              // ignored when alpha_beta_scheme != "none"
///         "Tc / K": ...,
///         "MW / g/mol": ...,                      // required when alpha_beta_scheme != "none"
///         "volume_shift / m^3/mol": ...,          // optional, default 0 (Péneloux T-independent shift)
///         "BibTeXKey": "...",
///         "(mu^*)^2": ...,    // optional polar
///         "nmu": ...,
///         "xp": ...,
///         "(Q^*)^2": ...,
///         "nQ": ...
///       },
///       ...
///     ]
///   }
///
/// When ``alpha_beta_scheme = "alajmi-2022"`` the user-supplied ``c1`` is
/// overwritten by Alajmi-Sisco 2022's MW correlation for kappa_alpha (along
/// with the kappa_beta covolume T-dependence). See ::alajmi_kappa.
///
/// The ``MW / g/mol`` field is the value the active MW-driven scheme will
/// see as input. For non-polymer species this is the molecular molar mass.
/// For polymer species the caller is responsible for supplying the *monomer*
/// (repeat-unit) molar mass instead.
inline auto CubicSAFTfactory(const nlohmann::json& spec) {
    CubicVariant variant = parse_cubic_variant(spec.at("cubic"));
    ChainRDF rdf = ChainRDF::Elliott;
    if (spec.contains("radial_dist")) {
        rdf = parse_chain_rdf(spec.at("radial_dist"));
    }
    AlphaBetaScheme scheme = AlphaBetaScheme::None;
    if (spec.contains("alpha_beta_scheme")) {
        scheme = parse_alpha_beta_scheme(spec.at("alpha_beta_scheme"));
    }
    double R_gas = spec.at("R_gas / J/mol/K");

    std::vector<CubicSAFTCoeffs> coeffs;
    for (auto& j : spec.at("coeffs")) {
        CubicSAFTCoeffs c;
        c.name = j.at("name");
        c.m    = j.at("m");
        c.a0i  = j.at("a0i / Pa m^6/mol^2");
        c.bi   = j.at("bi / m^3/mol");
        c.c1   = j.at("c1");
        c.Tc   = j.at("Tc / K");
        if (j.contains("MW / g/mol")) c.MW = j.at("MW / g/mol");
        if (j.contains("volume_shift / m^3/mol")) c.volume_shift = j.at("volume_shift / m^3/mol");
        c.BibTeXKey = j.value("BibTeXKey", std::string{});
        if (j.contains("epsilon_over_k")) c.epsilon_over_k = j.at("epsilon_over_k");
        if (j.contains("(mu^*)^2")) c.mustar2 = j.at("(mu^*)^2");
        if (j.contains("nmu"))      c.nmu     = j.at("nmu");
        if (j.contains("xp"))       c.xp      = j.at("xp");
        if (j.contains("(Q^*)^2"))  c.Qstar2  = j.at("(Q^*)^2");
        if (j.contains("nQ"))       c.nQ      = j.at("nQ");
        coeffs.push_back(c);
    }

    std::optional<Eigen::ArrayXXd> kmat;
    if (spec.contains("kmat") && spec.at("kmat").is_array() && spec.at("kmat").size() > 0) {
        auto raw = spec.at("kmat").get<std::vector<std::vector<double>>>();
        auto N = raw.size();
        Eigen::ArrayXXd km(N, N);
        for (std::size_t i = 0; i < N; ++i) {
            if (raw[i].size() != N) {
                throw teqp::InvalidArgument("CubicSAFT: kmat must be square");
            }
            for (std::size_t j = 0; j < N; ++j) {
                km(i, j) = raw[i][j];
            }
        }
        kmat = km;
    }

    return CubicSAFTNonpolarMixture(coeffs, variant, rdf, scheme, R_gas, kmat);
}

}  // namespace teqp::saft::cubicsaft
