# Smoke tests for renewal.jl
# Run with: julia --project src/test_renewal.jl

include("renewal.jl")

println("=== Smoke tests for renewal.jl ===\n")

# Test 1: Create a RenewalModel with default parameters
println("1. Creating RenewalModel with defaults...")
model = RenewalModel()
println("   ✓ da = $(model.da), amax = $(model.amax)")
println("   ✓ gen_interval = $(model.gen_interval)")
println("   ✓ obs_dist = $(model.obs_dist)")

# Test 2: Build model steps
println("\n2. Building model steps...")
h, A, gi = build_model_steps(model)
ages = model.da:model.da:model.amax
println("   ✓ ages: $(length(ages)) bins from $(first(ages)) to $(last(ages))")
println("   ✓ obs_hazards h: length=$(length(h)), sum=$(round(sum(h), digits=4))")
println("   ✓ gen_pmf gi: length=$(length(gi)), sum=$(round(sum(gi), digits=6))")
println("   ✓ A matrix: size=$(size(A))")

# Check hazards are non-negative
@assert all(h .>= 0) "Hazards should be non-negative"
println("   ✓ All hazards non-negative ✓")

# Check gi values are non-negative (it's a pdf evaluated at points, not normalized)
@assert all(gi .>= 0) "Generation interval values should be non-negative"
println("   ✓ All generation interval values non-negative ✓")

# Test 3: Build time stepper
println("\n3. Building time stepper...")
stepper = build_time_stepper(model; Ts=1.0)
println("   ✓ Time stepper created")

# Test 4: Run a single time step
println("\n4. Testing single time step...")
n_ages = length(ages)
x0 = vcat(fill(1.0, n_ages), [0.0])  # Initial state: 1 infected at each age, 0 cumulative obs
p = (Rt=1.5, pt=0.1)
x1 = stepper(x0, nothing, p, 0.0)
println("   ✓ State vector: $(length(x0)) → $(length(x1))")
println("   ✓ Initial total infected: $(round(sum(x0[1:end-1]), digits=2))")
println("   ✓ Next total infected: $(round(sum(x1[1:end-1]), digits=2))")
println("   ✓ Cumulative observations: $(round(x1[end], digits=4))")

# Test 5: Build particle filter dynamics
println("\n5. Building PF dynamics...")
pf_dyn = build_renewal_dyn_pf(model; Ts=1.0)
println("   ✓ PF dynamics function created")

# Test 6: Test PF dynamics (deterministic)
println("\n6. Testing PF dynamics (deterministic)...")
n_state = n_ages + 1 + 3  # ages + cumulative obs + log_Rt_mod + log_pt_mod + new_infs
x0_pf = vcat(fill(1.0, n_ages), [0.0], [0.0, 0.0, 0.0])
p_pf = (R0=1.5, p_obs=0.1, rho_p=0.9, sigma_p=0.1, rho_r=0.9, sigma_r=0.1)
x1_pf = pf_dyn(x0_pf, nothing, p_pf, 0.0, false)
println("   ✓ Deterministic step: $(length(x0_pf)) → $(length(x1_pf))")

# Test 7: Test PF dynamics (stochastic)
println("\n7. Testing PF dynamics (stochastic)...")
x1_pf_noisy = pf_dyn(x0_pf, nothing, p_pf, 0.0, true)
println("   ✓ Stochastic step completed")
println("   ✓ log_Rt_mod: $(round(x1_pf_noisy[end-1], digits=4))")
println("   ✓ log_pt_mod: $(round(x1_pf_noisy[end], digits=4))")

# Test 8: Measurement functions
println("\n8. Testing measurement functions...")
ll = renewal_meas_ll(x1_pf, nothing, 5, p_pf, 0.0)
println("   ✓ Log-likelihood for y=5: $(round(ll, digits=4))")

meas_det = renewal_meas_poi(x1_pf, nothing, p_pf, 0.0, false)
meas_noisy = renewal_meas_poi(x1_pf, nothing, p_pf, 0.0, true)
println("   ✓ Deterministic measurement: $(round(meas_det[1], digits=4))")
println("   ✓ Stochastic measurement: $(meas_noisy[1])")

println("\n=== All smoke tests passed! ===")
