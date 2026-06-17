#pragma once

#include "teqp/models/pcsaft.hpp"
#include "teqp/models/saftvrmie.hpp"
#include "teqp/models/association/association.hpp"
#include "teqp/models/saft/softsaft.hpp"
#include "teqp/models/saft/cubicsaft.hpp"
#include "teqp/models/model_potentials/2center_ljf.hpp"

namespace teqp::saft::genericsaft{

struct GenericSAFT{

public:
    using TwoCLJ = twocenterljf::Twocenterljf<twocenterljf::DipolarContribution>;
    using NonPolarTerms = std::variant<
        saft::pcsaft::PCSAFTMixture,
        SAFTVRMie::SAFTVRMieNonpolarMixture,
        saft::softsaft::SoftSAFT,
        saft::cubicsaft::CubicSAFTNonpolarMixture,
        TwoCLJ>;
//    using PolarTerms = EOSTermContainer<>;
    using AssociationTerms = std::variant<association::Association>;

private:
    auto make_nonpolar(const nlohmann::json &j) -> NonPolarTerms{
        std::string kind = j.at("kind");
        if (kind == "PCSAFT" || kind == "PC-SAFT"){
            return saft::pcsaft::PCSAFTfactory(j.at("model"));
        }
        else if (kind == "SAFTVRMie" || kind == "SAFT-VR-Mie"){
            return SAFTVRMie::SAFTVRMieNonpolarfactory(j.at("model"));
        }
        else if (kind == "Johnson+Johnson" || kind == "softSAFT"){
            return saft::softsaft::SoftSAFT(j.at("model"));
        }
        else if (kind == "cubic" || kind == "CubicSAFT" || kind == "CPCA"){
            return saft::cubicsaft::CubicSAFTfactory(j.at("model"));
        }
        else if (kind == "2CLJF" || kind == "2CLJ"){
            const auto& model = j.at("model");
            return twocenterljf::build_two_center_model(model.at("author"), model.at("L^*"));
        }
        else{
            throw std::invalid_argument("Not valid nonpolar kind:" + kind);
        }
    };
    auto make_association(const nlohmann::json &j) -> AssociationTerms{
        std::string kind = j.at("kind");
        if (kind == "canonical" || kind == "Dufal"){
            return association::Association::factory(j.at("model"));
        }
        else{
            throw std::invalid_argument("Not valid association kind:" + kind);
        }
    };
    
public:
    /// True when the nonpolar layer is a PC-SAFT mixture using the von Solms
    /// simplified hard-sphere term. Used to propagate the simplified g^hs into
    /// the association layer so a single toggle controls both.
    bool nonpolar_is_vonsolms_hs() const {
        return std::visit([](const auto& t) -> bool {
            if constexpr (std::is_same_v<std::decay_t<decltype(t)>, saft::pcsaft::PCSAFTMixture>) {
                return t.get_hard_sphere_variant() == saft::pcsaft::HardSphereVariant::Simplified;
            } else {
                return false;
            }
        }, nonpolar);
    }

    GenericSAFT(const nlohmann::json&j) : nonpolar(make_nonpolar(j.at("nonpolar"))){
        if (j.contains("association")){
            // von Solms consistency: when the hard-sphere term is the simplified
            // single-eta form, the association radial distribution g^hs must use
            // the SAME simplified contact value (g^hs flows to all consumers). 
            // Override the association block's radial_dist (which
            // lives under model.options, see Association::get_association_options)
            // to "vonSolms" unless the user pinned one explicitly.
            nlohmann::json assoc = j.at("association");
            if (nonpolar_is_vonsolms_hs() && assoc.contains("model")) {
                auto& amodel = assoc.at("model");
                const bool user_set =
                    amodel.contains("options")
                    && amodel.at("options").contains("radial_dist");
                if (!user_set) {
                    amodel["options"]["radial_dist"] = "vonSolms";
                }
            }
            association.emplace(make_association(assoc));
        }
    }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return get_R_gas<decltype(molefrac[0])>();
    }

    /// b_mix [m^3/mol] for the density solver. Association and polar layers
    /// are attractive corrections and do not change the η-mapping, so this
    /// delegates to the nonpolar (hard-chain) layer's get_bmix.
    template<typename VecType>
    double get_bmix(const double T, const VecType& molefracs) const {
        return std::visit([&](const auto& t) -> double {
            if constexpr (requires { t.get_bmix(T, molefracs); }) {
                return t.get_bmix(T, molefracs);
            } else {
                throw teqp::NotImplementedError(
                    "get_bmix is not implemented for this GenericSAFT nonpolar variant");
            }
        }, nonpolar);
    }

    /// Coarse model-family tag; GenericSAFT is a SAFT variant by construction.
    teqp::cppinterface::ModelKind get_model_kind() const {
        return teqp::cppinterface::ModelKind::SAFT;
    }

    NonPolarTerms nonpolar;
//    std::optional<PolarTerms> polar;
    std::optional<AssociationTerms> association;
    
    template <typename TType, typename RhoType, typename MoleFractions>
    auto alphar(const TType& T, const RhoType& rho, const MoleFractions& molefrac) const {
        auto contrib = std::visit([&](auto& t) { return t.alphar(T, rho, molefrac); }, nonpolar);
        if (association){
            const AssociationTerms& at = association.value();
            contrib += std::visit([&](auto& t) { return t.alphar(T, rho, molefrac); }, at);
        }
        return contrib;
    }
};

}
