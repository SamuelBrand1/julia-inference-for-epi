using SeeToDee, Random, Distributions, LinearAlgebra, Parameters
using LogExpFunctions
const rng = Random.Xoshiro()

@kwdef struct RenewalModel{D <: Distribution, G <: Distribution}
    da = 0.5                    # Infectious age
    amax = 30.0                 # Maximum infectious age
    gen_interval::G = Gamma(2.0, 3.0 / 2.0)  # Generation interval distribution
    obs_dist::D = Gamma(2.0, 7.0 / 2.0)  # Observation delay distribution
end

function make_obs_hazards(obs_dist::Distribution, ages, da)
    hazards = [pdf(obs_dist, a) / ccdf(obs_dist, a) for a in ages]
    hazards[isnan.(hazards)] .= 0.0  # Handle NaNs
    return hazards .* exp.(-cumsum(hazards) .* da)
end

function make_gen_pmf(gen_dist::Distribution, ages)
    pmf = pdf.(gen_dist, ages)
    return pmf
end

function build_int_mat(n::Int, da::Float64)
    Bidiagonal(fill(-inv(da), n), fill(inv(da), n-1), :L)
end

function build_model_steps(model::RenewalModel)
    ages = model.da:model.da:model.amax
    h = make_obs_hazards(model.obs_dist, ages, model.da)
    A = build_int_mat(length(ages), model.da)
    gi = make_gen_pmf(model.gen_interval, ages)
    return h, A, gi
end

function build_time_stepper(model::RenewalModel;
        integrator = SeeToDee.Rk4,
        Ts = 1.0,
        kwargs...)
    # Set up model for continuous time dynamics
    h, A, gi = build_model_steps(model)

    function tsi_ode(x, u, p, t)
        @unpack Rt, pt = p
        ut = @view x[1:(end - 1)]

        # Incidence and observation rate
        incidence = Rt * dot(ut, gi) * model.da
        obs_rate = pt * dot(h, ut) * model.da

        # time dynamics
        du = A * ut
        du[1] += incidence * inv(model.da)

        return vcat(du, [obs_rate])
    end

    return integrator(tsi_ode, Ts; kwargs...)
end

function build_renewal_dyn_pf(model::RenewalModel;
        integrator = SeeToDee.Rk4,
        Ts = 1.0,
        kwargs...)
    
    time_stepper = build_time_stepper(model; integrator = integrator, Ts = Ts, kwargs...)

    function renewal_dynamics(x, u, p, t, noise = false)
        if noise
            # unpack parameters
            @unpack R0, p_obs, rho_p, sigma_p, rho_r, sigma_r = p
            # unpack state
            x_state = @view x[1:(end - 3)]
            # modify Rt and pt with AR(1) process
            log_Rt_mod = x[end-2] * rho_r + sigma_r * randn(rng)
            log_pt_mod = x[end-1] * rho_p + sigma_p * randn(rng)
            # compute Rt and pt
            Rt = R0 * exp(log_Rt_mod)
            pt = logistic(logit(p_obs) + log_pt_mod)
            # record previous cumulative observations
            prev_cum_obs = x_state[end]
            # perform time step on continuous state
            next_x_state = time_stepper(x_state, u, (; Rt, pt), t)
            # record new cumulative observations with noise
            new_cum_obs = next_x_state[end] + 1e-3 * randn(rng)
            # record observed new infections
            new_infs = new_cum_obs - prev_cum_obs
            return vcat(next_x_state, [log_Rt_mod, log_pt_mod, new_infs])
        else
            @unpack R0, p_obs, rho_p, sigma_p, rho_r, sigma_r = p
            x_state = @view x[1:(end - 3)]
            log_Rt_mod = x[end-2]
            log_pt_mod = x[end-1]
            Rt = R0 * exp(log_Rt_mod)
            pt = logistic(logit(p_obs) + log_pt_mod)
            prev_cum_obs = x_state[end]
            next_x_state = time_stepper(x_state, u, (; Rt, pt), t)
            new_cum_obs = next_x_state[end] + 1e-3 * randn(rng)
            new_infs = new_cum_obs - prev_cum_obs
            return vcat(next_x_state, [log_Rt_mod, log_pt_mod, new_infs])
        end
    end
    return renewal_dynamics
end

function renewal_meas_ll(x, u, y, p, t)
    pred = max(x[end], 1e-3)
    logpdf(Poisson(pred), y[1])
end

function renewal_meas_poi(x, u, p, t, noise = false)
    pred = max(x[end], 1e-3)
    if noise
        return [rand(rng, Poisson(pred))]
    else
        return [pred]
    end
end
