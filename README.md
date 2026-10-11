# Triage of 300,000+ MOFs for CO₂ capture with machine learning

**The question.** A technology-scouting or R&D team wants to shortlist CO₂-capture sorbents from a library of 300,000+ candidate metal-organic frameworks (MOFs). Simulating every one is slow and costly. Can a cheap model, trained only on a MOF's *structure*, tell us which ones to simulate first, and how far can we trust it?

**The answer, in short**

- **Yes, for ranking.** Simulating only the model's top 10% captures **79% of the truly best 10%** of MOFs (7.9× better than random picking), stable across 5 random splits (79.1% ± 0.6%).
- **It transfers partially to unseen MOF families.** On held-out frameworks, absolute predictions drift, but rankings stay useful in most families: it finds 2× to 12× more of a family's best MOFs than random picking, with one clear failure (the *etb* topology).
- **It finds materials that are good on two measures at once.** Among 20 test MOFs with both high capacity (≥ 2 mmol/g) and high CO₂/N₂ selectivity (≥ 100), 17 sit in the model's top 5% by predicted capacity, even though the model never saw selectivity.

![enrichment](figures/enrichment.png)

## Data

Boyd et al., *Data-driven design of metal–organic frameworks for wet flue gas CO₂ capture*, Nature 576, 253–256 (2019). Dataset: Materials Cloud Archive, DOI [10.24435/materialscloud:2018.0016](https://doi.org/10.24435/materialscloud:2018.0016/v3) (CC BY 4.0). About 324,000 hypothetical MOFs with GCMC-simulated CO₂ and N₂ adsorption. The data is **not** included in this repo.

This project uses a random sample of **50,000 MOFs** from the full-database CSV, split 80/20 into train and test (about 10,000 test MOFs).

- **Target:** vacuum-swing working capacity (mmol/g): CO₂ taken up from flue gas (0.15 bar CO₂, 298 K) minus CO₂ remaining at regeneration conditions.
- **Inputs (structure only, 159 features):** volume, weight, surface area, void fraction, void volume, three pore-diameter measures, topology, metal node and two organic linkers (as categorical IDs), and functional groups (one flag per group).
- **Deliberately excluded:** every other column in the file (other uptakes, selectivity, heats of adsorption, etc.). They are simulation *results*, so using them would leak the answer.

## Method

1. Feature engineering from structural columns only.
2. Random Forest and XGBoost baselines on a random 80/20 split.
3. The XGBoost baseline repeated over 5 random splits to measure stability.
4. **Leave-one-topology-out** evaluation: train on all topologies except one, test on the held-out one, and report R², bias, rank correlation (Spearman) and enrichment.
5. **Enrichment**: how many of the truly best k% MOFs appear in the model's top k% predictions (random picking would find k%).
6. SHAP analysis of the XGBoost model.
7. Capacity vs. selectivity (Pareto) analysis and a candidate shortlist.

Everything is in one notebook, `mof_co2_screening.ipynb`, run on Google Colab.

## Results

### Prediction accuracy

| Model (single 80/20 split) | R² | MAE (mmol/g) |
|---|---|---|
| Random Forest | 0.824 | 0.110 |
| XGBoost | **0.856** | **0.100** |

XGBoost over 5 random splits: **R² = 0.859 ± 0.003, MAE = 0.099 ± 0.001 mmol/g** (mean ± std).

![parity](figures/parity.png)

Error grows at high capacity: the model tends to underpredict the best MOFs, a common effect of tree models on rare extreme values.

### Does it find the best MOFs? (enrichment, 5 random splits)

| Screen the model's top… | Share of the true top-k% found | Random baseline | Enrichment |
|---|---|---|---|
| 1% | 64.0% ± 1.7% | 1% | 64× |
| 5% | 72.8% ± 0.8% | 5% | 14.6× |
| 10% | 79.1% ± 0.6% | 10% | 7.9× |

### Does it transfer to unseen MOF families? (leave-one-topology-out, XGBoost)

| Held-out topology | n | R² | Spearman | Bias (pred − true) | True std | Found of true top 5% | Enrichment @5% | Enrichment @10% |
|---|---|---|---|---|---|---|---|---|
| pcu | 31,668 | 0.237 | 0.655 | +0.060 | 0.429 | 29.8% | 6.0× | 3.7× |
| nbo | 7,531 | −0.503 | 0.780 | +0.092 | 0.099 | 57.7% | 11.5× | 6.6× |
| sra | 3,708 | 0.463 | 0.762 | −0.119 | 0.728 | 44.3% | 8.9× | 5.0× |
| etb | 3,063 | −1.933 | 0.276 | +0.201 | 0.203 | 16.3% | 3.3× | 2.2× |
| acs | 2,157 | 0.367 | 0.751 | −0.200 | 0.554 | 23.4% | 4.7× | 3.6× |
| bcu | 1,288 | 0.421 | 0.910 | −0.243 | 0.575 | 21.9% | 4.4× | 4.5× |

Topologies with fewer than 300 MOFs (pts, fof, tbo, the) were not evaluated. Each row is a single run.

Transfer is **partial**. Absolute predictions shift by roughly ±0.1 to 0.24 mmol/g in unseen families, and R² is negative for two of them. But the picture is better than R² alone suggests:
- For **nbo**, the negative R² comes from a small upward offset against very little variation in the family (true std 0.099). Its within-family ranking is good (Spearman 0.78) and it has the highest enrichment (11.5× at 5%).
- **etb is the one real failure** (Spearman 0.28, 3.3× enrichment), though still above random.
- Rankings are good (Spearman 0.66 to 0.91) in five of six families. For screening, ranking matters more than absolute values.

### What the model uses (SHAP)

![shap](figures/shap_summary.png)

Predictions are driven mainly by **pore geometry**: MOFs with low void fraction and small pore diameters (but sufficient void volume per gram) are predicted to have the highest working capacity, consistent with strong CO₂ adsorption in narrow pores at low partial pressure. A few specific linkers (`organic_linker2` 29 and 18) give large positive contributions. Halogenated functional groups (F, I) lower predictions; for heavy halogens this may partly be a per-gram mass effect (a hypothesis, not tested). SHAP describes what the model relies on, not what causes performance, and correlated features (e.g. void fraction and void volume) share credit.

### Capacity vs. selectivity and the candidate shortlist

![pareto](figures/pareto.png)

The model ranks on capacity only, but still recovers candidates that are good on both measures. In the test set, 20 MOFs have capacity ≥ 2 mmol/g **and** CO₂/N₂ selectivity ≥ 100:
- **17 of the 20 fall within the model's top 5%** by predicted capacity (97.3rd to 99.9th percentile), and all 20 are within its top 11%.
- The three it ranked lower were underpredicted by about 1.2 to 1.8 mmol/g (percentiles 0.89 to 0.92).
- 13 of the 20 are `sra` frameworks with the `m5` metal node; the rest are `pcu` (4) and `acs` (3).

The black line is the Pareto front of simulated values, drawn only for MOFs with capacity ≥ 0.5 mmol/g (6 MOFs in the test set), because selectivity is a ratio of two simulated uptakes and is extremely noisy at near-zero capacity. It misses the high-selectivity/low-capacity end, as expected from a capacity-only model.

The model's top-10 predicted MOFs include 9 with simulated capacity above 2.3 mmol/g (dataset mean: 0.42 mmol/g), and one large miss (predicted 3.0, simulated 1.5). The ranked top-50 list is in `top_candidates.csv`.

## What this means in practice

**Suggested screening workflow (a proposal, not something validated here):**
1. Rank the full library with the model.
2. Run confirmatory simulations, including selectivity, on the top 5 to 10%.
3. Filter by what the model cannot see: water tolerance, stability, synthesisability, cost.
4. Experimentally validate a handful of the best.

**Why capacity matters for plant sizing (illustration).** A working capacity of 3 mmol/g is about 0.13 g CO₂ per g of sorbent (3 mmol × 44.01 g/mol), so roughly 7.6 g of sorbent must be cycled per g of CO₂ captured per cycle. At 1.5 mmol/g that doubles to about 15 g. All else equal, sorbent inventory scales inversely with working capacity, which drives bed size and capital cost. This is a back-of-envelope scaling, not a cost estimate.

## Limitations and risks

| Risk | Why it matters | Mitigation / next step |
|---|---|---|
| Labels are GCMC simulations on hypothetical structures | Force-field and partial-charge errors carry into the model; hypothetical MOFs may not be synthesisable | Validate top candidates with higher-level simulation and experiments |
| Dry flue gas at one condition (298 K); water not modelled | Humidity strongly affects real MOF performance | Add a humidity screen (the refined top-MOFs subset has water data) |
| Geometry + linker/topology IDs do not describe binding chemistry | Limits transfer to unseen topologies and linkers (etb fails) | Chemistry-aware descriptors or graph neural networks |
| Capacity-only model; selectivity is the simulated value | Pareto analysis is not model-only | Train a second (log-selectivity) model |
| No stability, synthesisability or cost information | "High predicted capacity" is not "good sorbent" | Add filters before any experimental spend |
| Single sample (50,000 MOFs) and one run per topology row | Topology results carry sampling noise (small families especially) | Repeat topology tests over seeds / larger samples |

## Future work

- Seed-averaged leave-one-topology-out results, and held-out-linker / held-out-metal-node splits.
- A second model for log-selectivity, to rank on both objectives from predictions alone.
- Feature ablation (geometry only → + linkers/topology → + functional groups).
- Benchmark against amine scrubbing (the incumbent) using sourced regeneration-energy figures.
- Humidity screening and regeneration-energy estimates using heat-of-adsorption data from the refined top-MOFs subset.

## Reproduce

```bash
pip install -r requirements.txt
```

1. Download `screening_data.tar.gz` from the Materials Cloud record above and extract it.
2. Open `mof_co2_screening.ipynb` in Google Colab (a GPU runtime speeds up XGBoost).
3. In section 1, set `file_path` to the location of the all-MOFs CSV (the notebook loads it from Google Drive), then run all cells.

## Repo structure

```
README.md
mof_co2_screening.ipynb   # full pipeline, with outputs
requirements.txt
top_candidates.csv        # ranked top-50 predicted MOFs (test set)
figures/                  # enrichment, parity, SHAP, Pareto plots
```

## Citation

Boyd, P. G. et al. Data-driven design of metal–organic frameworks for wet flue gas CO₂ capture. *Nature* 576, 253–256 (2019).
