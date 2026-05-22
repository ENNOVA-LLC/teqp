#pragma once
/**
 *  density_solver.hpp
 *
 *  Iterative volume-root solver for residual-Helmholtz models. Newton-Raphson
 *  on packing fraction eta in (0, 1) with a bisection safeguard, started from
 *  one or several heuristic eta seeds (vapor, liquid, dense).
 *
 *  Three entry points:
 *
 *    - solve_density(model, T, P, x, mode) -> double
 *        The common case. Returns the thermodynamically stable molar density
 *        at (T, P, x) by Gibbs-energy ranking across all converged roots.
 *
 *    - solve_density_roots(model, T, P, x, mode) -> vector<double>
 *        The diagnostic. Returns every converged root. Use for
 *        phase-stability inspection or when the caller wants to choose a
 *        non-stable root deliberately.
 *
 *    - solve_density_from_guess(model, T, P, x, rho_guess, mode) -> double
 *        Warm start from a user-supplied density guess. Falls back to the
 *        cold multi-seed solve if the warm Newton fails. Returns NaN only
 *        if both paths fail.
 *
 *  The solver does not assume a particular model kind (cubic vs SAFT vs
 *  multifluid); it queries the model for ``b_mix`` via ``get_bmix(T, x)``
 *  and for its family classifier via ``get_model_kind()``, so each model
 *  owns its own volume scale and seed-picking policy.
 *
 */

#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

#include "teqp/cpp/model_kind.hpp"

namespace teqp {
namespace density_solver {

constexpr double kEtaMin   = 1e-12;
constexpr double kEtaMax   = 0.9999;
constexpr double kTolH     = 1e-12;
constexpr double kTolEta   = 1e-13;
constexpr double kTolHFallback = 1e-6;
constexpr int    kMaxIter  = 100;
constexpr double R_GAS     = 8.314462618;
constexpr double kDupRel   = 1e-4;

/// Newton-on-eta with bisection safeguard. Returns NaN on failure.
template <typename Model>
double nr_eta_solve(const Model& model, double T,
                    const Eigen::ArrayXd& z,
                    double b_mix, double zb,
                    double eta_init)
{
    double eta;
    if      (eta_init < kEtaMin) eta = kEtaMin;
    else if (eta_init > kEtaMax) eta = kEtaMax;
    else                         eta = eta_init;

    double eta_lo = kEtaMin;
    double eta_hi = kEtaMax;

    for (int it = 0; it < kMaxIter; ++it) {
        if (eta < kEtaMin || eta > kEtaMax) {
            eta = 0.5 * (eta_lo + eta_hi);
        }

        const double rho = eta / b_mix;
        double Ar01, Ar02;
        try {
            Ar01 = model.get_Ar01(T, rho, z);
            Ar02 = model.get_Ar02(T, rho, z);
        } catch (...) {
            return std::nan("");
        }

        const double Z = 1.0 + Ar01;
        const double Z_eta = (eta > 1e-30) ? (Ar01 + Ar02) / eta : 0.0;

        const double h     = eta * Z - zb;
        const double h_eta = Z + eta * Z_eta;

        if (std::abs(h) < kTolH) return rho;

        if (h > 0.0) eta_hi = eta;
        else         eta_lo = eta;

        double eta_new;
        if (std::abs(h_eta) > 1e-30) {
            eta_new = eta - h / h_eta;
            if (!(eta_lo < eta_new && eta_new < eta_hi)) {
                eta_new = 0.5 * (eta_lo + eta_hi);
            }
        } else {
            eta_new = 0.5 * (eta_lo + eta_hi);
        }
        eta = eta_new;

        if ((eta_hi - eta_lo) < kTolEta) {
            const double rho_final = eta / b_mix;
            const double Ar01_f = model.get_Ar01(T, rho_final, z);
            const double h_f = eta * (1.0 + Ar01_f) - zb;
            return (std::abs(h_f) < kTolHFallback) ? rho_final : std::nan("");
        }
    }

    // Best-effort exit: accept if final residual is within fallback tolerance.
    const double rho = eta / b_mix;
    const double Ar01 = model.get_Ar01(T, rho, z);
    const double h = eta * (1.0 + Ar01) - zb;
    return (std::abs(h) < kTolHFallback) ? rho : std::nan("");
}

/// Build heuristic eta seeds for a given mode. ``zb = P*b/(R*T)`` is the
/// ideal-gas packing fraction at the spec; for a cubic vapor root,
/// ``eta ~ zb`` is the natural Newton starting point.
///
/// Seeds are specialized on the model family:
///
///   - Cubic: vapor seed tracks zb (the ideal-gas eta), liquid seed at 0.5.
///     "auto" mode does NOT include a dense (eta=0.95) root since cubics have
///     at most three Z-roots and the dense root is just a higher-eta version
///     of the same liquid branch.
///
///   - SAFT: vapor seed at a low fixed eta (1e-3) because the dispersive +
///     chain terms shift the vapor branch away from the cubic ideal-gas
///     scaling; liquid at 0.5; "auto" also tries dense at 0.95 (SAFT can have 
///     a separate ultra-dense liquid root in supercritical regions).
///
///   - Multifluid / IdealGas: not handled here; multifluid does not expose
///     b_mix so the solver throws upstream of seed_etas.
///
///   - Other (default for models that didn't opt in to get_model_kind):
///     try all three seeds under "auto" so a dense root is not missed.
inline std::vector<double> seed_etas(const std::string& mode,
                                     teqp::cppinterface::ModelKind kind,
                                     double zb)
{
    using teqp::cppinterface::ModelKind;
    const bool is_cubic = (kind == ModelKind::Cubic);
    const bool is_saft  = (kind == ModelKind::SAFT);

    auto vapor_seed = [&]() {
        if (is_cubic) {
            // Cubic: ideal-gas heuristic, clamped to (1e-12, 0.5).
            double v = zb;
            if (v < 1e-12) v = 1e-12;
            if (v > 0.5)   v = 0.5;
            return v;
        }
        // SAFT / Other: fixed low eta.
        return 1e-3;
    };

    if (mode == "vapor")   return {vapor_seed()};
    if (mode == "liquid")  return {0.5};
    if (mode == "dense")   return {0.95};
    if (mode == "liquid+") return {0.5, 0.95};
    if (mode == "auto") {
        if (is_cubic) return {vapor_seed(), 0.5};            // no dense for cubics
        if (is_saft)  return {vapor_seed(), 0.5, 0.95};      // SAFT may have dense root
        // Unknown / Other family: conservative -- try every seed.
        return {vapor_seed(), 0.5, 0.95};
    }
    throw std::invalid_argument("Unknown density-solve mode: " + mode);
}

/// Diagnostic entry point. Returns all converged roots in ascending order of discovery. 
/// Use this when you want to inspect the full root structure.
/// For the common case "return the thermodynamically stable density," use
/// solve_density() instead.
///
/// b_mix is queried from the model via model.get_bmix(T, x), so callers
/// never need to compute it themselves — the EOS owns its own volume scale.
template <typename Model>
std::vector<double> solve_density_roots(const Model& model,
                                        double T, double P,
                                        const Eigen::ArrayXd& z,
                                        const std::string& mode)
{
    double b_mix = model.get_bmix(T, z);
    if (b_mix <= 0.0) b_mix = 1e-10;
    const double zb = P * b_mix / (R_GAS * T);

    auto etas = seed_etas(mode, model.get_model_kind(), zb);

    std::vector<double> roots;
    roots.reserve(etas.size());

    for (double eta0 : etas) {
        const double rho = nr_eta_solve(model, T, z, b_mix, zb, eta0);
        if (!std::isnan(rho)) {
            bool dup = false;
            for (double r : roots) {
                if (std::abs(rho - r) / std::max(r, 1e-30) < kDupRel) {
                    dup = true;
                    break;
                }
            }
            if (!dup) roots.push_back(rho);
        }
    }
    return roots;
}

/// Pick the thermodynamically stable root (lowest reduced Gibbs) from a
/// list of converged densities at (T, x). For a single root this just
/// returns it.
template <typename Model>
double pick_stable_root(const Model& model, double T,
                        const Eigen::ArrayXd& z,
                        const std::vector<double>& roots)
{
    if (roots.empty()) return std::nan("");
    if (roots.size() == 1) return roots[0];

    double best_rho = roots[0];
    double best_g = std::numeric_limits<double>::infinity();
    for (double rho : roots) {
        const double Ar00 = model.get_Ar00(T, rho, z);
        const double Ar01 = model.get_Ar01(T, rho, z);
        const double g = Ar00 + Ar01 + std::log(rho);
        if (g < best_g) { best_g = g; best_rho = rho; }
    }
    return best_rho;
}

/// Main entry point. Returns the thermodynamically stable molar density
/// at (T, P, x). NaN if no root converged.
template <typename Model>
double solve_density(const Model& model,
                     double T, double P,
                     const Eigen::ArrayXd& z,
                     const std::string& mode)
{
    auto roots = solve_density_roots(model, T, P, z, mode);
    return pick_stable_root(model, T, z, roots);
}

/// Warm-start variant: the caller supplies a density guess (in mol/m^3),
/// typically the converged density from a previous outer-loop iteration in
/// a TP-flash. Runs one Newton from that guess; if it fails,
/// falls back automatically to the multi-seed cold solve in the supplied
/// ``mode`` and picks the Gibbs-stable root. The caller never has to retry
/// from Python on warm-start failure.
///
/// Pass mode="auto" to get the full vapor/liquid/dense sweep on fallback;
/// pass a narrower mode (e.g. "liquid") if the caller has prior knowledge
/// of the phase type.
template <typename Model>
double solve_density_from_guess(const Model& model,
                                double T, double P,
                                const Eigen::ArrayXd& z,
                                double rho_guess_mol_per_m3,
                                const std::string& mode)
{
    double b_mix = model.get_bmix(T, z);
    if (b_mix <= 0.0) b_mix = 1e-10;
    const double zb = P * b_mix / (R_GAS * T);
    const double eta0 = rho_guess_mol_per_m3 * b_mix;

    const double rho_warm = nr_eta_solve(model, T, z, b_mix, zb, eta0);
    if (!std::isnan(rho_warm)) return rho_warm;

    // Warm-start failed; fall back to cold multi-seed solve.
    return solve_density(model, T, P, z, mode);
}

} // namespace density_solver
} // namespace teqp
