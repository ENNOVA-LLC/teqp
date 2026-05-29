#pragma once

#include <Eigen/Dense>
#include "teqp/types.hpp"

namespace teqp::saft {

/**
 \brief Chen-Kreglewski temperature-dependent segment diameter.

 The temperature-dependent hard-sphere diameter used by PC-SAFT
 (Gross & Sadowski 2001) and by JC dipolar / BMCSL association layers
 that consume a SAFT segment diameter:

     d_i(T) = sigma_i * (1 - 0.12 * exp(-3 * eps_i / (k_B T)))

 The factor 0.12 and exponent -3 come from Chen & Kreglewski (1977)'s
 perturbation theory fit; see Gross-Sadowski 2001 Eq. A.9.

 Both ``sigma_i`` and the returned ``d_i`` carry the *same length unit*
 (whatever ``sigma_i`` is in -- Angstrom or meters); the function only
 scales the input. ``epsilon_over_k_i`` is in K and ``T`` is in K, so
 the exponent argument is dimensionless.
 */
template <typename TType, typename SigmaVec, typename EpsVec>
auto chen_kreglewski_d(const TType& T,
                        const SigmaVec& sigma,
                        const EpsVec& epsilon_over_k)
{
    using out_t = std::decay_t<TType>;
    Eigen::ArrayX<out_t> d(sigma.size());
    for (Eigen::Index i = 0; i < sigma.size(); ++i) {
        d[i] = sigma[i] * (1.0 - 0.12 * exp(-3.0 * epsilon_over_k[i] / T));
    }
    return d;
}

}  // namespace teqp::saft
