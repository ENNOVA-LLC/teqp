#pragma once

// This header defines pure enum / struct types used by the polar-term
// classes and the multipolar_contributions_variant. It is deliberately
// header-only with no external dependencies so it can be parsed in any
// translation unit without requiring nlohmann/json on the include path.
//
// The NLOHMANN_JSON_SERIALIZE_ENUM registrations for the two JSON-facing
// enums (multipolar_rhostar_approach, SigmaijRule) live in
// polar_terms.hpp, which is the canonical aggregator and explicitly
// includes nlohmann/json.hpp before invoking the macro.

namespace teqp::saft::polar_terms {

enum class multipolar_argument_spec {
    TK_rhoNA3_packingfraction_molefractions,
    TK_rhoNm3_rhostar_molefractions
};

enum class multipolar_rhostar_approach {
    kInvalid,
    use_packing_fraction,
    calculate_Gubbins_rhostar
};

/// Combining rule for the segment diameter sigma_ij in the polar
/// contribution.
/// - ``arithmetic`` is the stock GV / Lorentz-Berthelot convention;
/// - ``geometric`` is the Marshall-style sqrt(sigma_i * sigma_j) rule
///     which makes the A2 double sum and the A3 triple sum separable across components.
/// Pure-component results are unchanged by this choice (sigma_ii is the same in both rules);
/// mixture results shift when components have asymmetric size.
enum class SigmaijRule {
    arithmetic,
    geometric
};

template<typename type>
struct MultipolarContributionGubbinsTwuTermsGT{
    type alpha2;
    type alpha3;
    type alpha;
};

}
