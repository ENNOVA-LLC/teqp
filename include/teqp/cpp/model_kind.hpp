#pragma once
/**
 *  model_kind.hpp
 *
 *  Coarse family classifier for EOS models, used to specialize behavior
 *  (e.g. eta-seed selection in the density solver) without dynamic_cast
 *  or std::type_index lookups. Lives in its own header so concrete model
 *  classes can declare ``get_model_kind()`` without pulling in the full
 *  AbstractModel interface (teqpcpp.hpp).
 */

namespace teqp {
namespace cppinterface {

enum class ModelKind {
    Cubic,       ///< vdW / PR / SRK / advancedPRaEres / CPA cubic side
    SAFT,        ///< PC-SAFT, SAFT-VR-Mie, GenericSAFT, softSAFT, 2CLJF
    Multifluid,  ///< GERG2004/2008, multifluid reference, LJ126 etc.
    IdealGas,
    Other,
};

} // namespace cppinterface
} // namespace teqp
