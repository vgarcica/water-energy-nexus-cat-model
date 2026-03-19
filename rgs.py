# =============================================================================
# EXECUCIÓ
# =============================================================================
n_samples = 100

# Generar escenaris
escenaris = generar_escenaris(config, n_samples, seed=987654321, max_prealerta=MAX_PREALERTA)

# Executar en paral·lel
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

t0 = time.time()

results = Parallel(n_jobs=-1, verbose=0, max_nbytes=None)(
    delayed(run_case)(params) for params in tqdm(escenaris, desc="Escenaris")
)
print(f"Temps total: {time.time() - t0:.1f}s")
