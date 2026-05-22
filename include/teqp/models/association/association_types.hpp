#pragma once
#include "teqp/constants.hpp"

namespace teqp {
namespace association{

enum class association_classes {not_set, a1A, a2B, a3B, a4C, not_associating};

inline auto get_association_classes(const std::string& s) {
    if (s == "1A") { return association_classes::a1A; }
    else if (s == "2B") { return association_classes::a2B; }
    else if (s == "2B") { return association_classes::a2B; }
    else if (s == "3B") { return association_classes::a3B; }
    else if (s == "4C") { return association_classes::a4C; }
    else {
        throw std::invalid_argument("bad association flag: " + s);
    }
}

enum class radial_dists {
    CS,     ///< Carnahan-Starling at mixture eta built from T-independent sigma; cheap, used by CPA-style models.
    KG,     ///< Kontogeorgis form g = 1 / (1 - 1.9 eta); also from sigma.
    BMCSL,  ///< Boublik-Mansoori-Carnahan-Starling-Leland pair-specific g_ij(d_i(T), d_j(T)); the Gross-Sadowski 2002 PC-SAFT-association choice and feos's choice. Requires per-species (m, sigma, epsilon/k) in the CanonicalData.
};

inline auto get_radial_dist(const std::string& s) {
    if (s == "CS") { return radial_dists::CS; }
    else if (s == "KG") { return radial_dists::KG; }
    else if (s == "BMCSL") { return radial_dists::BMCSL; }
    else {
        throw std::invalid_argument("bad radial_dist flag: " + s);
    }
}

enum class Delta_rules {
    not_set,
    CR1,        ///< Wertheim CR1: arithmetic-mean b (==> arithmetic-mean m*sigma^3), geometric-mean beta, arithmetic-mean epsilon. teqp's historical convention.
    CR1_WS,     ///< CR1 with the Wolbach-Sandler 1998 cross-association combining rule:
                ///<   kappa_ij = sqrt(kappa_i * kappa_j),
                ///<   sigma_ij^3 = (sigma_i * sigma_j)^(3/2),  i.e. geometric mean of sigma^3 not arithmetic mean of m*sigma^3.
    Dufal,
};

inline auto get_Delta_rule(const std::string& s) {
    if (s == "CR1") { return Delta_rules::CR1; }
    else if (s == "CR1-WS" || s == "CR1_WS") { return Delta_rules::CR1_WS; }
    else if (s == "Dufal") { return Delta_rules::Dufal; }
    else {
        throw std::invalid_argument("bad Delta_rule flag: " + s);
    }
}

struct CanonicalData{
    Eigen::ArrayXd b_m3mol, ///< The covolume b, in m^3/mol, one per component
        beta, ///< The volume factor, dimensionless, one per component
        epsilon_Jmol; ///< The association energy of each molecule, in J/mol, one per component
    radial_dists radial_dist;

    // Per-species PC-SAFT segment parameters. Required when
    // ``radial_dist == BMCSL`` so the Delta routine can build d_i(T) and the
    // BMCSL pair g_ij(d_i, d_j).
    Eigen::ArrayXd m_segments;          ///< Chain length m_i [-]
    Eigen::ArrayXd sigma_m;             ///< Segment diameter sigma_i [m]
    Eigen::ArrayXd epsilon_over_k_K;    ///< Segment energy parameter eps_i/k_B [K]
};

struct DufalData{
    // Parameters coming from the non-associating part, one per component
    Eigen::ArrayXd sigma_m, epsilon_Jmol, lambda_r;
    Eigen::ArrayXXd kmat; ///< Matrix of k_{ij} values
    
    // Parameters from the associating part, one per component
    Eigen::ArrayXd epsilon_HB_Jmol, K_HB_m3;
    
    Eigen::ArrayXd SIGMA3ij_m3, EPSILONOVERKij_K, LAMBDA_Rij, EPSILONOVERK_HBij_K, KHBij_m3;
    
    void apply_mixing_rules(){
        std::size_t N = sigma_m.size();
        SIGMA3ij_m3.resize(N,N);
        EPSILONOVERKij_K.resize(N,N);
        LAMBDA_Rij.resize(N,N);
        EPSILONOVERK_HBij_K.resize(N,N);
        KHBij_m3.resize(N,N);
        
        for (auto i = 0U; i < N; ++i){
            for (auto j = 0U; j < N; ++j){
                SIGMA3ij_m3(i,j) = POW3((sigma_m[i] + sigma_m[j])/2.0);
                EPSILONOVERKij_K(i, j) = (1-kmat(i,j))*sqrt(POW3(sigma_m[i]*sigma_m[j]))/SIGMA3ij_m3(i,j)*sqrt(epsilon_Jmol[i]*epsilon_Jmol[j])/constants::R_CODATA2017;
                LAMBDA_Rij(i, j) = 3 + sqrt((lambda_r[i]-3)*(lambda_r[j]-3));
                EPSILONOVERK_HBij_K(i, j) = sqrt(epsilon_HB_Jmol[i]*epsilon_HB_Jmol[j])/constants::R_CODATA2017;
                KHBij_m3(i,j) = POW3((cbrt(K_HB_m3[i]) + cbrt(K_HB_m3[j]))/2.0); // Published erratum in Dufal: https://doi.org/10.1080/00268976.2017.1402604
            }
        }
    }
};


}
}
