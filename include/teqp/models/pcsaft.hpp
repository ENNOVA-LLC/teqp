/***
 
 \brief This file contains the contributions that can be composed together to form SAFT models

*/

#pragma once

#include "nlohmann/json.hpp"
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/constants.hpp"
#include "teqp/json_tools.hpp"
#include "teqp/models/saft/pcsaftpure.hpp"
#include "teqp/models/saft/polar_terms/GrossVrabec.hpp"
#include "teqp/models/saft/polar_terms/JogChapman.hpp"
#include "teqp/models/saft/segment_diameter.hpp"
#include "teqp/models/saft/peneloux_shift.hpp"
#include "teqp/cpp/model_kind.hpp"
#include <optional>

// Definitions for the matrices of global constants for the PCSAFT model
namespace teqp::saft::PCSAFT::PCSAFTMatrices{
namespace GrossSadowski2001{
    extern const Eigen::Array<double, 3, 7> a, b;
}
namespace LiangIECR2012{
    extern const Eigen::Array<double, 3, 7> a, b;
}
namespace LiangIECR2014{
    extern const Eigen::Array<double, 3, 7> a, b;
}
}

namespace teqp::saft::pcsaft {

//#define PCSAFTDEBUG

/// Coefficients for one fluid
struct SAFTCoeffs {
    std::string name; ///< Name of fluid
    double m = -1, ///< number of segments
        sigma_Angstrom = -1, ///< [A] segment diameter
        epsilon_over_k = -1; ///< [K] depth of pair potential divided by Boltzman constant
    std::string BibTeXKey; ///< The BibTeXKey for the reference for these coefficients
    double mustar2 = 0, ///< non-dimensional, the reduced dipole moment squared
           nmu = 0, ///< number of dipolar segments (Gross-Vrabec convention)
           xp = 0, ///< fraction of dipolar segments on the chain (Jog-Chapman convention; xp = nmu/m for consistency)
           Qstar2 = 0, ///< non-dimensional, the reduced quadrupole squared
           nQ = 0; ///< number of quadrupolar segments
    double volume_shift = 0; ///< Péneloux T-independent volume shift [m^3/mol]; default 0 (unshifted). See PCSAFTMixture::alphar.
};

/// Manager class for PCSAFT coefficients
class PCSAFTLibrary {
    std::map<std::string, SAFTCoeffs> coeffs;
public:
    PCSAFTLibrary() {
        insert_normal_fluid("Methane", 1.0000, 3.7039, 150.03, "Gross-IECR-2001");
        insert_normal_fluid("Ethane", 1.6069, 3.5206, 191.42, "Gross-IECR-2001");
        insert_normal_fluid("Propane", 2.0020, 3.6184, 208.11, "Gross-IECR-2001");
    }
    void insert_normal_fluid(const std::string& name, double m, const double sigma_Angstrom, const double epsilon_over_k, const std::string& BibTeXKey) {
        SAFTCoeffs coeff;
        coeff.name = name;
        coeff.m = m;
        coeff.sigma_Angstrom = sigma_Angstrom;
        coeff.epsilon_over_k = epsilon_over_k;
        coeff.BibTeXKey = BibTeXKey;
        coeffs.insert(std::pair<std::string, SAFTCoeffs>(name, coeff));
    }
    const auto& get_normal_fluid(const std::string& name) {
        auto it = coeffs.find(name);
        if (it != coeffs.end()) {
            return it->second;
        }
        else {
            throw std::invalid_argument("Bad name:" + name);
        }
    }
    auto get_coeffs(const std::vector<std::string>& names){
        std::vector<SAFTCoeffs> c;
        for (auto n : names){
            c.push_back(get_normal_fluid(n));
        }
        return c;
    }
};

/// Eqn. A.11
/// Erratum: should actually be 1/RHS of equation A.11 according to sample
/// FORTRAN code
template <typename Eta, typename Mbar>
auto C1(const Eta& eta, const Mbar& mbar) {
    auto oneeta = 1.0 - eta;
    return forceeval(1.0 / (1.0
        + mbar * (8.0 * eta - 2.0 * eta * eta) / (oneeta*oneeta*oneeta*oneeta)
        + (1.0 - mbar) * (20.0 * eta - 27.0 * eta * eta + 12.0 * eta*eta*eta - 2.0 * eta*eta*eta*eta) / ((1.0 - eta) * (2.0 - eta)*(1.0 - eta) * (2.0 - eta))));
}
/// Eqn. A.31
template <typename Eta, typename Mbar>
auto C2(const Eta& eta, const Mbar& mbar) {
    return forceeval(-pow(C1(eta, mbar), 2) * (
        mbar * (-4.0 * eta * eta + 20.0 * eta + 8.0) / pow(1.0 - eta, 5)
        + (1.0 - mbar) * (2.0 * eta * eta * eta + 12.0 * eta * eta - 48.0 * eta + 40.0) / pow((1.0 - eta) * (2.0 - eta), 3)
        ));
}

/// Residual contribution to alphar from hard-sphere (Eqn. A.6)
template<typename VecType, typename VecType2>
auto get_alphar_hs(const VecType& zeta, const VecType2& D) {
    /*
    The limit of alphar_hs in the case of density going to zero is zero,
    but its derivatives must still match so that the automatic differentiation tooling
    will work properly, so a Taylor series around rho=0 is constructed. The first term is
    needed for calculations of virial coefficient temperature derivatives.
    The term zeta_0 in the denominator is zero, but *ratios* of zeta values are ok because
    they cancel the rho (in the limit at least) so we can write that zeta_x/zeta_y = D_x/D_x where
    D_i = sum_i x_im_id_{ii}. This allows for the substitution into the series expansion terms.
    
    <sympy>
     from sympy import *
     zeta_0, zeta_1, zeta_2, zeta_3, rho = symbols('zeta_0, zeta_1, zeta_2, zeta_3, rho')
     D_0, D_1, D_2, D_3 = symbols('D_0, D_1, D_2, D_3')
     POW2 = lambda x: x**2
     POW3 = lambda x: x**3
     alpha = 1/zeta_0*(3*zeta_1*zeta_2/(1-zeta_3) + zeta_2**3/(zeta_3*POW2(1-zeta_3)) + (POW3(zeta_2)/POW2(zeta_3)-zeta_0)*log(1-zeta_3))
     alpha = alpha.subs(zeta_0, rho*D_0).subs(zeta_1, rho*D_1).subs(zeta_2, rho*D_2).subs(zeta_3, rho*D_3)
     for Nderiv in [1, 2, 3, 4, 5]:
         display(simplify((simplify(diff(alpha, rho, Nderiv)).subs(rho,0)*rho**Nderiv/factorial(Nderiv)).subs(D_0, zeta_0/rho).subs(D_1, zeta_1/rho).subs(D_2, zeta_2/rho).subs(D_3, zeta_3/rho)))
    </sympy>
     
     The updated approach is simpler; start off with the expression for alphar_hs, and simplify
     ratios where 0/0 for which the l'hopital limit would be ok until you remove all the terms
     in the denominator, allowing it to be evaluated. All terms have a zeta_x/zeta_0 and a few
     have zeta_3*zeta_0 in the denominator
     
    */
    auto Upsilon = 1.0 - zeta[3];
    if (getbaseval(zeta[3]) == 0){
        return forceeval(
             3.0*D[1]/D[0]*zeta[2]/Upsilon
             + D[2]*D[2]*zeta[2]/(D[3]*D[0]*Upsilon*Upsilon)
             - log(Upsilon)
             + (D[2]*D[2]*D[2])/(D[3]*D[3]*D[0])*log(Upsilon)
        );
    }
    
    return forceeval(1.0 / zeta[0] * (3.0 * zeta[1] * zeta[2] / Upsilon
        + zeta[2] * zeta[2] * zeta[2] / zeta[3] / Upsilon / Upsilon
        + (zeta[2] * zeta[2] * zeta[2] / (zeta[3] * zeta[3]) - zeta[0]) * log(1.0 - zeta[3])
        ));
}

/// Term from Eqn. A.7
template<typename zVecType, typename dVecType>
auto gij_HS(const zVecType& zeta, const dVecType& d,
    std::size_t i, std::size_t j) {
    auto Upsilon = 1.0 - zeta[3];
#if defined(PCSAFTDEBUG)
    auto term1 = forceeval(1.0 / (Upsilon));
    auto term2 = forceeval(d[i] * d[j] / (d[i] + d[j]) * 3.0 * zeta[2] / pow(Upsilon, 2));
    auto term3 = forceeval(pow(d[i] * d[j] / (d[i] + d[j]), 2) * 2.0 * zeta[2]*zeta[2] / pow(Upsilon, 3));
#endif
    return forceeval(1.0 / (Upsilon)+d[i] * d[j] / (d[i] + d[j]) * 3.0 * zeta[2] / pow(Upsilon, 2)
        + pow(d[i] * d[j] / (d[i] + d[j]), 2) * 2.0 * zeta[2]*zeta[2] / pow(Upsilon, 3));
}

/**
Sum up three array-like objects that can each have different container types and value types
*/
template<typename VecType1, typename NType>
auto powvec(const VecType1& v1, NType n) {
    auto o = v1;
    for (auto i = 0; i < v1.size(); ++i) {
        o[i] = pow(v1[i], n);
    }
    return o;
}

/**
Sum up the coefficient-wise product of three array-like objects that can each have different container types and value types
*/
template<typename VecType1, typename VecType2, typename VecType3>
auto sumproduct(const VecType1& v1, const VecType2& v2, const VecType3& v3) {
    using ResultType = typename std::common_type_t<decltype(v1[0]), decltype(v2[0]), decltype(v3[0])>;
    return forceeval((v1.template cast<ResultType>().array() * v2.template cast<ResultType>().array() * v3.template cast<ResultType>().array()).sum());
}

/***
 * \brief This class provides the evaluation of the hard chain contribution from classic PC-SAFT
 */
class PCSAFTHardChainContribution{

protected:
    const Eigen::ArrayX<double> m, ///< number of segments
        mminus1, ///< m-1
        sigma_Angstrom, ///<
        epsilon_over_k; ///< depth of pair potential divided by Boltzmann constant
    const Eigen::ArrayXXd kmat; ///< binary interaction parameter matrix
    Eigen::Array<double, 3, 7> a, ///< The universal constants used in Eqn. A.18 of G&S
                            b; ///< The universal constants used in Eqn. A.19 of G&S
    const teqp::saft::polar_terms::SigmaijRule sigmaij_rule;
        ///< Combining rule for sigma_ij inside the dispersion double sum.
        ///< - ``arithmetic`` (default) reproduces stock PC-SAFT (Lorentz-Berthelot);
        ///< - ``geometric`` is the Marshall convention sqrt(sigma_i*sigma_j).
        ///< Pure-component results are unchanged; mixture results shift when
        ///< components have asymmetric size.

public:
    PCSAFTHardChainContribution(const Eigen::ArrayX<double> &m, const Eigen::ArrayX<double> &mminus1, const Eigen::ArrayX<double> &sigma_Angstrom, const Eigen::ArrayX<double> &epsilon_over_k, const Eigen::ArrayXXd &kmat, const Eigen::Array<double, 3, 7>&a, const Eigen::Array<double, 3,7>&b, teqp::saft::polar_terms::SigmaijRule sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic)
    : m(m), mminus1(mminus1), sigma_Angstrom(sigma_Angstrom), epsilon_over_k(epsilon_over_k), kmat(kmat), a(a), b(b), sigmaij_rule(sigmaij_rule) {}

    PCSAFTHardChainContribution& operator=( const PCSAFTHardChainContribution& ) = delete; // non copyable

    /// Combining rule for sigma_ij in the dispersion accumulators.
    /// Pure-components (i==j) collapse to sigma_i in both rules.
    inline double pair_sigma(std::size_t i, std::size_t j) const {
        if (sigmaij_rule == teqp::saft::polar_terms::SigmaijRule::geometric) {
            return sqrt(sigma_Angstrom[i] * sigma_Angstrom[j]);
        }
        return 0.5 * (sigma_Angstrom[i] + sigma_Angstrom[j]);
    }

    template<typename TTYPE, typename RhoType, typename VecType>
    auto eval(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        
        Eigen::Index N = m.size();
        
        if (mole_fractions.size() != N) {
            throw std::invalid_argument("Length of mole_fractions (" + std::to_string(mole_fractions.size()) + ") is not the length of components (" + std::to_string(N) + ")");
        }
        
        using TRHOType = std::common_type_t<std::decay_t<TTYPE>, std::decay_t<RhoType>, std::decay_t<decltype(mole_fractions[0])>, std::decay_t<decltype(m[0])>>;
        
        // Chen-Kreglewski T-dependent segment diameter [A]
        Eigen::ArrayX<TTYPE> d = teqp::saft::chen_kreglewski_d(T, sigma_Angstrom, epsilon_over_k);
        TRHOType m2_epsilon_sigma3_bar = 0.0;
        TRHOType m2_epsilon2_sigma3_bar = 0.0;
        if (sigmaij_rule == teqp::saft::polar_terms::SigmaijRule::geometric) {
            // Collapsed form (Marshall): with sigma_ij = sqrt(sigma_i*sigma_j),
            // sigma_ij^3 = sigma_i^{3/2} sigma_j^{3/2} separates over i and j,
            // so the BIP-free part of each accumulator reduces to a perfect
            // square of a single O(N) sum. The BIP correction terms are
            // genuinely O(N^2), but they iterate only over nonzero kij in
            // practice (fast when kij == 0).
            //
            // Define per-component scalars:
            //   u_i = x_i m_i sqrt(eps_i / T) sigma_i^{3/2}
            //   v_i = x_i m_i (eps_i / T)     sigma_i^{3/2}
            // Then (BIP-free):
            //   m2_eps_sig3   = (sum u_i)^2
            //   m2_eps2_sig3  = (sum v_i)^2
            // With BIPs, expanding (1-k)^p:
            //   m2_eps_sig3   = (sum u_i)^2 - sum_ij u_i u_j k_ij
            //   m2_eps2_sig3  = (sum v_i)^2 - 2 sum_ij v_i v_j k_ij
            //                                 + sum_ij v_i v_j k_ij^2
            Eigen::ArrayX<TRHOType> u(N), v(N);
            for (auto i = 0L; i < N; ++i) {
                auto sigma_pow_3_2 = sigma_Angstrom[i] * sqrt(sigma_Angstrom[i]);
                auto ekT_i = epsilon_over_k[i] / T;
                auto common = mole_fractions[i] * m[i] * sigma_pow_3_2;
                u[i] = common * sqrt(ekT_i);
                v[i] = common * ekT_i;
            }
            TRHOType U = u.sum();
            TRHOType V = v.sum();
            m2_epsilon_sigma3_bar = U * U;
            m2_epsilon2_sigma3_bar = V * V;
            // BIP correction (only nonzero entries contribute).Iterate
            // the full N^2 to avoid maintaining a separate sparse index;
            // the early-skip below makes this effectively O(nnz).
            for (auto i = 0L; i < N; ++i) {
                for (auto j = 0L; j < N; ++j) {
                    double k_ij = kmat(i, j);
                    if (k_ij == 0.0) continue;
                    auto uiuj = u[i] * u[j];
                    auto vivj = v[i] * v[j];
                    m2_epsilon_sigma3_bar  -= uiuj * k_ij;
                    m2_epsilon2_sigma3_bar += vivj * (-2.0 * k_ij + k_ij * k_ij);
                }
            }
        }
        else {
            // Stock arithmetic (Lorentz-Berthelot) path: full N^2 loop with
            // sigma_ij = (sigma_i + sigma_j) / 2 (does not separate).
            for (auto i = 0L; i < N; ++i) {
                for (auto j = 0; j < N; ++j) {
                    // Eq. A.5 of Gross-Sadowski 2001
                    auto sigma_ij = pair_sigma(i, j);
                    auto eij_over_k = sqrt(epsilon_over_k[i] * epsilon_over_k[j]) * (1.0 - kmat(i,j));
                    auto sigmaij3 = sigma_ij*sigma_ij*sigma_ij;
                    auto ekT = eij_over_k/T;
                    m2_epsilon_sigma3_bar  += mole_fractions[i] * mole_fractions[j] * m[i] * m[j] * ekT       * sigmaij3;
                    m2_epsilon2_sigma3_bar += mole_fractions[i] * mole_fractions[j] * m[i] * m[j] * (ekT*ekT) * sigmaij3;
                }
            }
        }
        auto mbar = (mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()).sum();
        
        /// Convert from molar density to number density in molecules/Angstrom^3
        RhoType rho_A3 = rhomolar * N_A * 1e-30; //[molecules (not moles)/A^3]
        
        constexpr double MY_PI = EIGEN_PI;
        double pi6 = (MY_PI / 6.0);
        
        /// Evaluate the components of zeta
        using ta = std::common_type_t<decltype(m[0]), decltype(d[0]), decltype(rho_A3)>;
        std::vector<ta> zeta(4), D(4);
        for (std::size_t n = 0; n < 4; ++n) {
            // Eqn A.8
            auto dn = pow(d, static_cast<int>(n));
            TRHOType xmdn = forceeval((mole_fractions.template cast<TRHOType>().array()*m.template cast<TRHOType>().array()*dn.template cast<TRHOType>().array()).sum());
            D[n] = forceeval(pi6*xmdn);
            zeta[n] = forceeval(D[n]*rho_A3);
        }
        
        /// Packing fraction is the 4-th value in zeta, at index 3
        auto eta = zeta[3];
        
        Eigen::Array<decltype(eta), 7, 1> etapowers; etapowers(0) = 1.0; for (auto i = 1U; i <= 6; ++i){ etapowers(i) = eta*etapowers(i-1); }
        using TYPE = TRHOType;
        Eigen::Array<TYPE, 7, 1> abar = (a.row(0).cast<TYPE>().array() + ((mbar - 1.0) / mbar) * a.row(1).cast<TYPE>().array() + ((mbar - 1.0) / mbar) * ((mbar - 2.0) / mbar) * a.row(2).cast<TYPE>().array()).eval();
        Eigen::Array<TYPE, 7, 1> bbar = (b.row(0).cast<TYPE>().array() + ((mbar - 1.0) / mbar) * b.row(1).cast<TYPE>().array() + ((mbar - 1.0) / mbar) * ((mbar - 2.0) / mbar) * b.row(2).cast<TYPE>().array()).eval();
        auto I1 = (abar.array().template cast<decltype(eta)>()*etapowers).sum();
        auto I2 = (bbar.array().template cast<decltype(eta)>()*etapowers).sum();
        
        // Hard chain contribution from G&S
        using tt = std::common_type_t<decltype(zeta[0]), decltype(d[0])>;
        Eigen::ArrayX<tt> lngii_hs(mole_fractions.size());
        for (auto i = 0; i < lngii_hs.size(); ++i) {
            lngii_hs[i] = log(gij_HS(zeta, d, i, i));
        }
        auto alphar_hc = forceeval(mbar * get_alphar_hs(zeta, D) - sumproduct(mole_fractions, mminus1, lngii_hs)); // Eq. A.4
        
        // Dispersive contribution
        auto C1_ = C1(eta, mbar);
        auto alphar_disp = forceeval(-2 * MY_PI * rho_A3 * I1 * m2_epsilon_sigma3_bar - MY_PI * rho_A3 * mbar * C1_ * I2 * m2_epsilon2_sigma3_bar);
                    
#if defined(PCSAFTDEBUG)
        if (!std::isfinite(getbaseval(alphar_hc))){
            throw teqp::InvalidValue("An invalid value was obtained for alphar_hc; please investigate");
        }
        if (!std::isfinite(getbaseval(I1))){
            throw teqp::InvalidValue("An invalid value was obtained for I1; please investigate");
        }
        if (!std::isfinite(getbaseval(I2))){
            throw teqp::InvalidValue("An invalid value was obtained for I2; please investigate");
        }
        if (!std::isfinite(getbaseval(C1_))){
            throw teqp::InvalidValue("An invalid value was obtained for C1; please investigate");
        }
        if (!std::isfinite(getbaseval(alphar_disp))){
            throw teqp::InvalidValue("An invalid value was obtained for alphar_disp; please investigate");
        }
#endif
        struct PCSAFTHardChainContributionTerms{
            TRHOType eta;
            TRHOType alphar_hc;
            TRHOType alphar_disp;
        };
        return PCSAFTHardChainContributionTerms{eta, alphar_hc, alphar_disp};
    }
};

/** A class used to evaluate mixtures using PC-SAFT model

This is the classical Gross and Sadowski model from 2001: https://doi.org/10.1021/ie0003887
 
with the errors fixed as noted in a comment: https://doi.org/10.1021/acs.iecr.9b01515
*/
/// Polar-theory selector.  Default GrossVrabec preserves the historical
/// behavior of PCSAFTMixture; JogChapman activates the Dominik/Jog-Chapman
/// dipolar contribution implemented in JogChapman.hpp.
enum class PolarModel { GrossVrabec, JogChapman };

class PCSAFTMixture {
public:
    using PCSAFTDipolarContribution = teqp::saft::polar_terms::GrossVrabec::DipolarContributionGrossVrabec;
    using PCSAFTDipolarContributionJC = teqp::saft::polar_terms::JogChapman::DipolarContributionJogChapman;
    using PCSAFTQuadrupolarContribution = teqp::saft::polar_terms::GrossVrabec::QuadrupolarContributionGross;
protected:
    Eigen::ArrayX<double> m, ///< number of segments
        mminus1, ///< m-1
        sigma_Angstrom, ///<
        epsilon_over_k, ///< depth of pair potential divided by Boltzmann constant
        volume_shift; ///< per-species Péneloux T-independent volume shift [m^3/mol]; zero by default
    std::vector<std::string> names, bibtex;
    Eigen::ArrayXXd kmat; ///< binary interaction parameter matrix
    PolarModel polar_model = PolarModel::GrossVrabec;
    teqp::saft::polar_terms::SigmaijRule polar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic;
    teqp::saft::polar_terms::SigmaijRule nonpolar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic;
        ///< sigma_ij rule inside the non-polar dispersion double sum.
        ///< - 'arithmetic' (default) preserves stock PC-SAFT behavior; 
        ///< - 'geometric' is the Marshall convention. Independent of polar_sigmaij_rule because
        ///< the two layers have different parameter calibrations.

    PCSAFTHardChainContribution hardchain;
    std::optional<PCSAFTDipolarContribution> dipolar; // GV dipolar, can be present or not
    std::optional<PCSAFTDipolarContributionJC> dipolar_jc; // JC dipolar, alternative to dipolar
    std::optional<PCSAFTQuadrupolarContribution> quadrupolar; // Can be present or not

    void check_kmat(Eigen::Index N) {
        if (kmat.cols() != kmat.rows()) {
            throw teqp::InvalidArgument("kmat rows and columns are not identical");
        }
        if (kmat.cols() == 0) {
            kmat.resize(N, N); kmat.setZero();
        }
        else if (kmat.cols() != N) {
            throw teqp::InvalidArgument("kmat needs to be a square matrix the same size as the number of components");
        }
    };
    auto get_coeffs_from_names(const std::vector<std::string> &the_names){
        PCSAFTLibrary library;
        return library.get_coeffs(the_names);
    }
    auto build_hardchain(const std::vector<SAFTCoeffs> &coeffs, const Eigen::Array<double, 3, 7>& a, const Eigen::Array<double, 3, 7>& b){
        check_kmat(coeffs.size());

        m.resize(coeffs.size());
        mminus1.resize(coeffs.size());
        sigma_Angstrom.resize(coeffs.size());
        epsilon_over_k.resize(coeffs.size());
        volume_shift.resize(coeffs.size());
        names.resize(coeffs.size());
        bibtex.resize(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            m[i] = coeff.m;
            mminus1[i] = m[i] - 1;
            sigma_Angstrom[i] = coeff.sigma_Angstrom;
            epsilon_over_k[i] = coeff.epsilon_over_k;
            volume_shift[i] = coeff.volume_shift;
            names[i] = coeff.name;
            bibtex[i] = coeff.BibTeXKey;
            i++;
        }
        return PCSAFTHardChainContribution(m, mminus1, sigma_Angstrom, epsilon_over_k, kmat, a, b, nonpolar_sigmaij_rule);
    }
    auto extract_names(const std::vector<SAFTCoeffs> &coeffs){
        std::vector<std::string> names_;
        for (const auto& c: coeffs){
            names_.push_back(c.name);
        }
        return names_;
    }
    auto build_dipolar(const std::vector<SAFTCoeffs> &coeffs) -> std::optional<PCSAFTDipolarContribution>{
        // Only build the GV dipolar contribution when polar_model == GrossVrabec.
        // For JogChapman, dipolar stays std::nullopt and dipolar_jc is built instead.
        if (polar_model != PolarModel::GrossVrabec) {
            return std::nullopt;
        }
        Eigen::ArrayXd mustar2(coeffs.size()), nmu(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            mustar2[i] = coeff.mustar2;
            nmu[i] = coeff.nmu;
            i++;
        }
        if ((mustar2*nmu).cwiseAbs().sum() == 0){
            return std::nullopt; // No dipolar contribution is present
        }
        // The dispersive and hard chain initialization has already happened at this point
        return PCSAFTDipolarContribution(m, sigma_Angstrom, epsilon_over_k, mustar2, nmu, polar_sigmaij_rule);
    }
    /**
     * \brief Build the Jog-Chapman dipolar contribution from the PCSAFT
     *        coefficient table.
     *
     * \details The PCSAFT JSON schema carries the dipole as the reduced
     * dimensionless ``(mu^*)^2`` from Gross-Vrabec 2006 (shared input
     * across teqp's polar layers):
     * \f[
     *    (\mu^*)^2 = \frac{\mu^2}{4 \pi \epsilon_0 \, m \, (\epsilon/k_B) \, k_B \, \sigma^3}
     * \f]
     * with \f$\epsilon/k_B\f$ in K, \f$\sigma\f$ in m, and \f$\mu\f$ in C·m.
     * The published JC kernel works in raw \f$\mu^2\f$ (C\f$^2\f$·m\f$^2\f$),
     * so we invert the reduction here once at construction:
     * \f[
     *    \mu_{SI}^2 = (\mu^*)^2 \cdot 4 \pi \epsilon_0 \, m \, (\epsilon/k_B) \, k_B \, \sigma^3
     * \f]
     * (single \f$k_B\f$, not \f$k_B^2\f$, because \f$\epsilon/k_B\f$ is the
     * dimensionless input).
     *
     * The polar-segment fraction ``xp`` falls back to ``nmu/m`` when not
     * supplied directly; see ``docs/eos/polar_theory_mapping.md`` for the
     * JC ↔ GV parameter mapping.
     */
    auto build_dipolar_jc(const std::vector<SAFTCoeffs> &coeffs) -> std::optional<PCSAFTDipolarContributionJC>{
        if (polar_model != PolarModel::JogChapman) {
            return std::nullopt;
        }
        Eigen::ArrayXd mustar2(coeffs.size()), xp(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            mustar2[i] = coeff.mustar2;
            xp[i] = (coeff.xp > 0) ? coeff.xp
                                   : ((coeff.m > 0 && coeff.nmu > 0) ? coeff.nmu / coeff.m : 0.0);
            i++;
        }
        if ((mustar2 * xp).cwiseAbs().sum() == 0) {
            return std::nullopt; // No dipolar contribution is present
        }
        constexpr double FOUR_PI_EPS0 = 1.11265005605362e-10;
        constexpr double K_B = 1.380649e-23;
        Eigen::ArrayXd mu_squared_SI(coeffs.size());
        for (Eigen::Index k = 0; k < (Eigen::Index)coeffs.size(); ++k) {
            const double sigma_m = sigma_Angstrom[k] * 1e-10;
            const double sigma3 = sigma_m * sigma_m * sigma_m;
            // Invert teqp's (mu*)^2 reduction; see docstring above.
            mu_squared_SI[k] = mustar2[k] * FOUR_PI_EPS0 * m[k] * epsilon_over_k[k]
                                * K_B * sigma3;
        }
        return PCSAFTDipolarContributionJC(m, xp, mu_squared_SI, polar_sigmaij_rule);
    }
    auto build_quadrupolar(const std::vector<SAFTCoeffs> &coeffs) -> std::optional<PCSAFTQuadrupolarContribution>{
        // The dispersive and hard chain initialization has already happened at this point
        Eigen::ArrayXd Qstar2(coeffs.size()), nQ(coeffs.size());
        auto i = 0;
        for (const auto &coeff : coeffs) {
            Qstar2[i] = coeff.Qstar2;
            nQ[i] = coeff.nQ;
            i++;
        }
        if ((Qstar2*nQ).cwiseAbs().sum() == 0){
            return std::nullopt; // No quadrupolar contribution is present
        }
        return PCSAFTQuadrupolarContribution(m, sigma_Angstrom, epsilon_over_k, Qstar2, nQ, polar_sigmaij_rule);
    }
    /**
     * \brief Pick the polar \f$\sigma_{ij}\f$ combining rule when the caller
     *        did not pin it explicitly.
     *
     * \details Per-polar-model defaults:
     * - JogChapman: ``geometric`` (Marshall convention; the O(N) collapse of
     *   JC's pair sums requires it).
     * - GrossVrabec: ``arithmetic`` (Lorentz-Berthelot, the historical
     *   behavior; GV's pair-resolved kernel supports either rule).
     *
     * An explicit ``override_rule`` always wins.
     */
    static teqp::saft::polar_terms::SigmaijRule resolve_polar_sigmaij_rule(
        PolarModel pm,
        std::optional<teqp::saft::polar_terms::SigmaijRule> override_rule)
    {
        if (override_rule.has_value()) {
            return override_rule.value();
        }
        return (pm == PolarModel::JogChapman)
            ? teqp::saft::polar_terms::SigmaijRule::geometric
            : teqp::saft::polar_terms::SigmaijRule::arithmetic;
    }
public:
    PCSAFTMixture(const std::vector<std::string> &names,
                  const Eigen::Array<double, 3, 7>& a = teqp::saft::PCSAFT::PCSAFTMatrices::GrossSadowski2001::a,
                  const Eigen::Array<double, 3, 7>& b = teqp::saft::PCSAFT::PCSAFTMatrices::GrossSadowski2001::b,
                  const Eigen::ArrayXXd& kmat = {},
                  PolarModel polar_model = PolarModel::GrossVrabec,
                  std::optional<teqp::saft::polar_terms::SigmaijRule> polar_sigmaij_rule = std::nullopt,
                  teqp::saft::polar_terms::SigmaijRule nonpolar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic)
        : PCSAFTMixture(get_coeffs_from_names(names), a, b, kmat, polar_model, polar_sigmaij_rule, nonpolar_sigmaij_rule){};
    PCSAFTMixture(const std::vector<SAFTCoeffs> &coeffs,
                  const Eigen::Array<double, 3, 7>& a = teqp::saft::PCSAFT::PCSAFTMatrices::GrossSadowski2001::a,
                  const Eigen::Array<double, 3, 7>& b = teqp::saft::PCSAFT::PCSAFTMatrices::GrossSadowski2001::b,
                  const Eigen::ArrayXXd &kmat = {},
                  PolarModel polar_model = PolarModel::GrossVrabec,
                  std::optional<teqp::saft::polar_terms::SigmaijRule> polar_sigmaij_rule = std::nullopt,
                  teqp::saft::polar_terms::SigmaijRule nonpolar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic)
        : names(extract_names(coeffs)), kmat(kmat),
          polar_model(polar_model),
          polar_sigmaij_rule(resolve_polar_sigmaij_rule(polar_model, polar_sigmaij_rule)),
          nonpolar_sigmaij_rule(nonpolar_sigmaij_rule),
          hardchain(build_hardchain(coeffs, a, b)),
          dipolar(build_dipolar(coeffs)),
          dipolar_jc(build_dipolar_jc(coeffs)),
          quadrupolar(build_quadrupolar(coeffs)) {};
    
//    PCSAFTMixture( const PCSAFTMixture& ) = delete; // non construction-copyable
    PCSAFTMixture& operator=( const PCSAFTMixture& ) = delete; // non copyable
    
    auto get_m() const { return m; }
    auto get_sigma_Angstrom() const { return sigma_Angstrom; }
    auto get_epsilon_over_k_K() const { return epsilon_over_k; }
    auto get_kmat() const { return kmat; }
    auto get_names() const { return names;}
    auto get_BibTeXKeys() const { return bibtex;}

    auto print_info() {
        std::string s = std::string("i m sigma / A e/kB / K \n  ++++++++++++++") + "\n";
        for (auto i = 0; i < m.size(); ++i) {
            s += std::to_string(i) + " " + std::to_string(m[i]) + " " + std::to_string(sigma_Angstrom[i]) + " " + std::to_string(epsilon_over_k[i]) + "\n";
        }
        return s;
    }
    
    template<typename VecType>
    double max_rhoN(const double T, const VecType& mole_fractions) const {
        Eigen::ArrayX<double> d = teqp::saft::chen_kreglewski_d(T, sigma_Angstrom, epsilon_over_k);
        return 6 * 0.74 / EIGEN_PI / (mole_fractions*m*powvec(d, 3)).sum()*1e30; // particles/m^3
    }

    /// Mixture hard-chain volume [m^3/mol] used by the C++ density solver
    /// to non-dimensionalize the volume root as eta = b_mix * rho.
    ///   b_mix = (pi/6) * N_A * sum_i x_i m_i d_i(T)^3
    /// with d_i(T) the Chen-Kreglewski diameter (see segment_diameter.hpp).
    /// d_i is in Angstrom inside this class; the 1e-30 converts the sum to m^3.
    template<typename VecType>
    double get_bmix(const double T, const VecType& mole_fractions) const {
        Eigen::ArrayX<double> d = teqp::saft::chen_kreglewski_d(T, sigma_Angstrom, epsilon_over_k);
        return (EIGEN_PI / 6.0) * N_A
               * (mole_fractions * m * powvec(d, 3)).sum() * 1e-30;
    }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    /// Coarse model-family tag
    teqp::cppinterface::ModelKind get_model_kind() const {
        return teqp::cppinterface::ModelKind::SAFT;
    }

    /**
     * \brief Residual molar Helmholtz energy with Péneloux volume translation.
     *
     * \details Per Moine et al. 2019 (I-PC-SAFT) and Palma et al. 2018, the
     * user-supplied ``rhomolar`` is the *shifted* (observable) molar
     * density. The unshifted PC-SAFT terms (hardchain, dispersion, polar,
     * quadrupolar) are evaluated at the shifted density
     * \f$\tilde\rho = \rho / (1 + \rho \bar c)\f$ with
     * \f$\bar c = \sum_i x_i c_i\f$ (Palma 2018 Eq. 3, linear-in-x), then
     * closed with \f$-\ln(1 + \rho \bar c)\f$ at the user's \f$\rho\f$.
     * Unlike cubic-SAFT, PC-SAFT has no explicit covolume \f$b\f$, so no
     * \f$b_t = b_o - c\f$ analogue is needed; \f$m\f$, \f$\sigma\f$, and
     * \f$\epsilon/k\f$ are unchanged by the translation. When every
     * species has \f$c_i = 0\f$ (default), this reduces bit-exactly to the
     * unshifted PC-SAFT. See peneloux_shift.hpp.
     */
    template<typename TTYPE, typename RhoType, typename VecType>
    auto alphar(const TTYPE& T, const RhoType& rhomolar, const VecType& mole_fractions) const {
        // Péneloux: evaluate sub-terms at rho_tilde, close with -ln(1+rho*cbar).
        auto rho_tilde = teqp::peneloux::shifted_density(
            rhomolar, mole_fractions, volume_shift);

        // First values for the chain with dispersion (always included).
        auto vals = hardchain.eval(T, rho_tilde, mole_fractions);
        auto alphar = forceeval(vals.alphar_hc + vals.alphar_disp);

        auto rho_A3 = forceeval(rho_tilde*N_A*1e-30);
        // GV dipolar (active when polar_model == GrossVrabec)
        if (dipolar){
            auto valsdip = dipolar.value().eval(T, rho_A3, vals.eta, mole_fractions);
            alphar += valsdip.alpha;
        }
        // JC dipolar: pass the CK temperature-dependent diameter explicitly
        // (JogChapman.hpp requires the caller to supply d).
        if (dipolar_jc){
            auto d_Angstrom = teqp::saft::chen_kreglewski_d(T, sigma_Angstrom, epsilon_over_k);
            auto valsdip = dipolar_jc.value().eval(T, rho_A3, vals.eta, mole_fractions, d_Angstrom);
            alphar += valsdip.alpha;
        }
        // If quadrupole is present, add its contribution
        if (quadrupolar){
            auto valsquad = quadrupolar.value().eval(T, rho_A3, vals.eta, mole_fractions);
            alphar += valsquad.alpha;
        }

        // Péneloux closing correction (zero by default).
        alphar += teqp::peneloux::log_correction(
            rhomolar, mole_fractions, volume_shift);

        return forceeval(alphar);
    }
};

/// A JSON-based factory function for the PC-SAFT model
inline auto PCSAFTfactory(const nlohmann::json& spec) {
    std::optional<Eigen::ArrayXXd> kmat;
    if (spec.contains("kmat") && spec.at("kmat").is_array() && spec.at("kmat").size() > 0){
        kmat = build_square_matrix(spec["kmat"]);
    }
    // By default use the a & b matrices of Gross&Sadowski, IECR, 2001
    Eigen::Array<double, 3, 7> a = teqp::saft::PCSAFT::PCSAFTMatrices::GrossSadowski2001::a,
    b = teqp::saft::PCSAFT::PCSAFTMatrices::GrossSadowski2001::b;
    // Unless overwritten by user selection via the "ab" field
    if (spec.contains("ab")){
        std::string source = spec.at("ab");
        if (source == "Liang-IECR-2012"){
            a = teqp::saft::PCSAFT::PCSAFTMatrices::LiangIECR2012::a;
            b = teqp::saft::PCSAFT::PCSAFTMatrices::LiangIECR2012::b;
        }
        else if (source == "Liang-IECR-2014"){
            a = teqp::saft::PCSAFT::PCSAFTMatrices::LiangIECR2014::a;
            b = teqp::saft::PCSAFT::PCSAFTMatrices::LiangIECR2014::b;
        }
        else{
            throw teqp::InvalidArgument("Don't know what to do with this source for a&b: " + source);
        }
    }
    
    // Optional polar-model selector. Defaults to "GrossVrabec" for backward
    // compatibility. Set "JogChapman" to activate the Dominik/Jog-Chapman
    // dipolar contribution implemented in JogChapman.hpp.
    PolarModel polar_model = PolarModel::GrossVrabec;
    if (spec.contains("polar_model")) {
        std::string pm = spec.at("polar_model");
        if (pm == "GrossVrabec" || pm == "gross-vrabec" || pm == "GV") {
            polar_model = PolarModel::GrossVrabec;
        } else if (pm == "JogChapman" || pm == "jog-chapman" || pm == "JC") {
            polar_model = PolarModel::JogChapman;
        } else {
            throw teqp::InvalidArgument("Don't know what to do with polar_model = " + pm
                + " (expected 'GrossVrabec' or 'JogChapman')");
        }
    }

    // Optional polar sigma_ij combining-rule selector. When omitted, the
    // PCSAFTMixture constructor picks: arithmetic for GV (Lorentz-Berthelot,
    // historical behavior) and geometric for JC (matches JC dipolar's
    // hard-coded geometric d_ij so the polar branches are internally
    // consistent). Set explicitly to override.
    std::optional<teqp::saft::polar_terms::SigmaijRule> polar_sigmaij_rule = std::nullopt;
    if (spec.contains("polar_combining_rule")) {
        std::string rule = spec.at("polar_combining_rule");
        if (rule == "arithmetic" || rule == "Lorentz-Berthelot" || rule == "LB") {
            polar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic;
        } else if (rule == "geometric" || rule == "Marshall") {
            polar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::geometric;
        } else {
            throw teqp::InvalidArgument("Don't know what to do with polar_combining_rule = " + rule
                + " (expected 'arithmetic' or 'geometric')");
        }
    }

    // Optional nonpolar sigma_ij combining-rule selector for the dispersion
    // double sum in PCSAFTHardChainContribution.
    teqp::saft::polar_terms::SigmaijRule nonpolar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic;
    if (spec.contains("nonpolar_combining_rule")) {
        std::string rule = spec.at("nonpolar_combining_rule");
        if (rule == "arithmetic" || rule == "Lorentz-Berthelot" || rule == "LB") {
            nonpolar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::arithmetic;
        } else if (rule == "geometric" || rule == "Marshall") {
            nonpolar_sigmaij_rule = teqp::saft::polar_terms::SigmaijRule::geometric;
        } else {
            throw teqp::InvalidArgument("Don't know what to do with nonpolar_combining_rule = " + rule
                + " (expected 'arithmetic' or 'geometric')");
        }
    }

    if (spec.contains("names")){
        std::vector<std::string> names = spec["names"];
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != names.size()){
            throw teqp::InvalidArgument("Provided length of names of " + std::to_string(names.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
        }
        return PCSAFTMixture(names, a, b, kmat.value_or(Eigen::ArrayXXd{}), polar_model, polar_sigmaij_rule, nonpolar_sigmaij_rule);
    }
    else if (spec.contains("coeffs")){
        std::vector<SAFTCoeffs> coeffs;
        // Debye to C*m conversion: 1 D = 3.33564e-30 C*m (SI).
        // Used to fold Marshall's lumped alpha_p [D^2] into teqp's
        // dimensionless (mu*)^2.
        constexpr double DEBYE_TO_CM = 3.33564e-30;
        for (auto j : spec["coeffs"]) {
            SAFTCoeffs c;
            c.name = j.at("name");
            c.m = j.at("m");
            c.sigma_Angstrom = j.at("sigma_Angstrom");
            c.epsilon_over_k = j.at("epsilon_over_k");
            c.BibTeXKey = j.at("BibTeXKey");
            if (j.contains("(mu^*)^2") && j.contains("nmu")){
                c.mustar2 = j.at("(mu^*)^2");
                c.nmu = j.at("nmu");
            }
            // JC's xp (fraction of polar segments) overrides nmu/m if both are present.
            if (j.contains("xp")) {
                c.xp = j.at("xp");
                // If only xp + (mu^*)^2 were supplied (no nmu), still want mustar2 picked up.
                if (j.contains("(mu^*)^2") && !j.contains("nmu")) {
                    c.mustar2 = j.at("(mu^*)^2");
                }
            }
            // Marshall ``alpha_p`` shorthand: lumped polar strength in [D^2].
            // Fold to dimensionless ``(mu*)^2`` via the JC<->GV invariant
            //   n_mu * mu_GV^2 = m * alpha_p,  i.e.,
            //   mu_eff^2 [D^2] = m * alpha_p / n_mu = alpha_p / xp.
            // Then convert to teqp's reduced form:
            //   (mu*)^2 = mu_eff^2 / (4*pi*eps0 * m * eps_k * k_B * sigma_m^3).
            // Only consumed when the user did NOT already provide (mu^*)^2.
            if (j.contains("alpha_p") && !j.contains("(mu^*)^2")) {
                double alpha_p_D2 = j.at("alpha_p");
                if (alpha_p_D2 > 0) {
                    // Pick n_mu: prefer explicit ``nmu``, else derive from xp*m,
                    // else fall back to m (i.e., assume every segment polar).
                    double n_mu_local;
                    if (j.contains("nmu")) {
                        n_mu_local = j.at("nmu");
                        c.nmu = n_mu_local;
                    } else if (j.contains("xp")) {
                        n_mu_local = c.m * static_cast<double>(j.at("xp"));
                        c.nmu = n_mu_local;
                    } else {
                        n_mu_local = c.m;
                        c.nmu = n_mu_local;
                    }
                    if (n_mu_local > 0) {
                        double mu_eff_D2 = c.m * alpha_p_D2 / n_mu_local;
                        double mu_eff_Cm2 = mu_eff_D2 * DEBYE_TO_CM * DEBYE_TO_CM;
                        double sigma_m = c.sigma_Angstrom * 1e-10;
                        double denom = 4.0 * static_cast<double>(EIGEN_PI)
                                       * teqp::constants::epsilon_0
                                       * c.m * c.epsilon_over_k
                                       * teqp::constants::k_B
                                       * sigma_m * sigma_m * sigma_m;
                        if (denom > 0) {
                            c.mustar2 = mu_eff_Cm2 / denom;
                        }
                    }
                }
            }
            if (j.contains("(Q^*)^2") && j.contains("nQ")){
                c.Qstar2 = j.at("(Q^*)^2");
                c.nQ = j.at("nQ");
            }
            if (j.contains("volume_shift / m^3/mol")){
                c.volume_shift = j.at("volume_shift / m^3/mol");
            }
            coeffs.push_back(c);
        }
        if (kmat && static_cast<std::size_t>(kmat.value().rows()) != coeffs.size()){
            throw teqp::InvalidArgument("Provided length of coeffs of " + std::to_string(coeffs.size()) + " does not match the dimension of the kmat of " + std::to_string(kmat.value().rows()));
        }
        return PCSAFTMixture(coeffs, a, b, kmat.value_or(Eigen::ArrayXXd{}), polar_model, polar_sigmaij_rule, nonpolar_sigmaij_rule);
    }
    else{
        throw std::invalid_argument("you must provide names or coeffs, but not both");
    }
}
using saft::PCSAFT::PCSAFTPureGrossSadowski2001;

}; // namespace teqp::saft

namespace teqp::PCSAFT{
using namespace teqp::saft::pcsaft;
}
