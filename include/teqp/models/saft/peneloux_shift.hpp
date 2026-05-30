#pragma once

/**
 * \file peneloux_shift.hpp
 * \brief Péneloux volume translation as a closed-form Helmholtz correction.
 *
 * \details The Péneloux (1982) volume shift maps the "corrected"
 * molar volume to the unshifted EOS's molar volume via
 * \f[
 *     V_\text{unshifted} = V_\text{user} + N \, \bar c(x), \qquad
 *     \bar c(x) = \sum_i x_i \, c_i
 * \f]
 *
 * Only the T-independent shift is supported here. A T-linear coefficient
 * (sometimes denoted \f$c_1\f$ in legacy literature, with
 * \f$c_i(T) = c_{0,i} + c_{1,i} T\f$) is deliberately **not** accepted:
 * a T-dependent \f$c\f$ breaks K-value temperature-derivative invariance
 * (see "K-value invariance" below).
 *
 * \par Sign convention (USER-FACING)
 * Positive \f$c_i > 0\f$ **lowers** the predicted molar volume (raises the
 * predicted density). This matches every standard Péneloux source
 * (Péneloux 1982 Eq. 8, Jaubert 2016, Privat-Jaubert 2018): fitted values
 * of \f$c_i\f$ are usually positive for cubic EOSs because they underpredict
 * liquid density, and the shift moves the predicted V downward to match
 * experiment.
 *
 * Pressure is invariant under the translation, which fixes the residual
 * Helmholtz transformation. With \f$\rho = N/V_\text{user}\f$ and
 * \f$\tilde\rho = N/V_\text{unshifted} = \rho / (1 + \rho \bar c)\f$:
 * \f[
 *     \alpha^r_\text{shifted}(T, \rho, x) =
 *     \alpha^r_\text{unshifted}(T, \tilde\rho, x) - \ln(1 + \rho \bar c)
 * \f]
 *
 * Recipe for any EOS aggregator that already has an unshifted
 * \f$\alpha^r(T, \rho, x)\f$ implementation: evaluate the underlying terms
 * at \f$\tilde\rho\f$ via ::shifted_density, then add ::log_correction.
 *
 * \par K-value invariance
 * The Péneloux correction adds a species-only term
 * \f$-c_i \cdot P / (RT)\f$ to \f$\ln \phi_i\f$ (sign matches the
 * convention above; a positive \f$c_i\f$ at positive \f$P\f$ lowers
 * \f$\ln \phi_i\f$). In \f$\ln K_i = \ln \phi_i^L - \ln \phi_i^V\f$,
 * both phases are evaluated at the same \f$(T, P)\f$, so the species
 * correction cancels exactly: \f$\ln K_i\f$ is invariant under any choice
 * of \f$c_i\f$, including the partial T-derivative
 * \f$\partial \ln K_i / \partial T\f$ (since \f$c_i\f$ is T-independent).
 *
 * \par Defaults
 * \f$c_i = 0\f$ for all species recovers the unshifted EOS bit-exactly:
 * \f$\bar c = 0 \Rightarrow \tilde\rho = \rho\f$ and
 * \f$-\ln(1 + 0) = 0\f$.
 *
 * Reference: Péneloux, Rauzy, Fréze (1982), *Fluid Phase Equilibria* 8, 7-23.
 */

#include <Eigen/Dense>
#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"

namespace teqp::peneloux {

/// Mole-fraction-averaged volume shift \f$\bar c(x) = \sum_i x_i c_i\f$.
/// T-independent (the only supported form).
template<typename VecType>
auto cbar(const VecType& mole_fractions, const Eigen::ArrayXd& c) {
    using sum_t = std::decay_t<decltype(mole_fractions[0])>;
    sum_t cb = 0.0;
    for (Eigen::Index i = 0; i < c.size(); ++i) {
        cb += mole_fractions[i] * c[i];
    }
    return forceeval(cb);
}

/// Shifted density \f$\tilde\rho = \rho / (1 + \rho \bar c)\f$ at which the
/// unshifted EOS must be evaluated. Positive \f$\bar c\f$ lowers
/// \f$\tilde\rho\f$ (user's denser state maps to a less-dense unshifted
/// state), which is the textbook Péneloux convention.
template<typename RhoType, typename VecType>
auto shifted_density(const RhoType& rho,
                      const VecType& mole_fractions,
                      const Eigen::ArrayXd& c) {
    auto cb = cbar(mole_fractions, c);
    return forceeval(rho / (1.0 + rho * cb));
}

/// Logarithmic Helmholtz correction term \f$-\ln(1 + \rho \bar c)\f$. Added
/// to the unshifted \f$\alpha^r(T, \tilde\rho, x)\f$ to obtain the shifted
/// \f$\alpha^r(T, \rho, x)\f$.
template<typename RhoType, typename VecType>
auto log_correction(const RhoType& rho,
                     const VecType& mole_fractions,
                     const Eigen::ArrayXd& c) {
    auto cb = cbar(mole_fractions, c);
    return forceeval(-log(1.0 + rho * cb));
}

} // namespace teqp::peneloux
