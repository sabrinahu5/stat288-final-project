# Stat 288 Final Project

Source city: Delhi, India
- Why Delhi? Lots of data

Target city: Lagos, Nigeria
- Why Lagos? No sensor data, but poor air quality, health issues

Google Earth Engine Data: 
- Choose high value bands from AOD (Main aerosol signal + quality + water‑vapour helps scaling), NDVI (Vegetation → dust suppression / seasonal burn), night-lights (Proxy for traffic & industry activity)
- Grab weekly composites
- Data cleaning:
    - Missing some bands for nightlight data: filter out images or add missing bands with a masked value

OpenAQ:
- Pull PM2.5 from Delhi and Lagos with a "bounding box", aggregate to weekly averages.
- First, use bounding box to pull locations in the area. Then, use locations to get sensor ids, then, use sensor ids to collect measurement data about PM2.5 levels.
- Data cleaning!!! get rid of outliers and missigness -- how should we think about this? (don't want to introduce bias)
    - Hampel Filters: Kills single‑point spikes without harming true multi‑hour events
    - Global Tail Trimming: Removes weeks with instrument drift or stuck sensors
    - Weekly Aggregation: Ensures each week is well‑supported

Google Earth Engine Monitors -> Earth Enginer FeatureCollection
Add long & lat band to merge with OpenAQ dataset

Then, merge images with labels. 

Problem with initial plan (transfer model), train just on Delhi, fine-tune on Lagos: 
- A LOT OF DATA BEING LOST WHEN MERGING --- disjoint!!!

Updated plan:
- Pick a big, Lagos-like source city, find a city with lots of weekly PM₂.₅ + patch data whose overall pollution regime (e.g. mean, variance) is closer to Lagos than Delhi is. BANGKOK!! similar climate, lots of data
- Train on Delhi + new city:
    - Build one big TF-record dataset of patches + pm25_mean across all weeks 2018–2021. Shuffle and split:
    - Train on 2018–20
    - Val on 2021
    - Test on 2022–23
    - Train your full CNN → dense‐head regression there.
- Evaluate zero-shot on Accra
    - Hold out Accra 2023 entirely for evaluation:
    - Zero-shot: feed Accra 2023 patches through your frozen CNN+head → predict pm25_mean → compute RMSE / R² / AUROC. This gives the raw transfer gap.
- Simulate label-scarcity on Accra
    - Now, on Accra 2018–2022, sub-sample N = {200, 100, 50, 20, 10} weekly labels:
    - For each N:
        - Fine-tune only your dense head (CNN’s first two conv blocks frozen) on N Accra examples (use sample_weight=5 to “amplify” each). LR = 1e-4, up to 10 epochs with early stopping on a small val split.
        - Test on held-out Accra 2023 → RMSE / ΔRMSE relative to zero-shot.
        - Plot RMSE vs. N. This curve tells you “with only N labels, I get this error bar.”
- Decide your Lagos strategy
    - From the Accra results we'll know:
    - Minimum N for “acceptable” RMSE (say < 20 µg/m³).
    - Whether bias-correction (linear α+β·yDelhi) ever beats fine-tuning when N is tiny.
    - Then, on Lagos:
    - Zero-shot Lagos 2023 → report RMSE₀.
    - If Lagos labels ≥ Nₘᵢₙ, fine-tune head on all Lagos 2018–2022 (or whatever you have) → report RMSE₁.
    - If Lagos labels < Nₘᵢₙ, switch to bias-correction.

Analysis:

Compare:
- Transfer model
- Local-only model (trained on target data only)
- Source-only model applied directly
Use RMSE, R², or visual comparison (scatter plots, residuals)

- Which model performed best?
- Which features were most important?
- Gain from transfer learning?

Visualizations:
- PM2.5 prediction vs. ground truth (scatter plot)
- Feature importance bar chart
- Map of source/target regions
- Performance comparison table

Paper Structure:
- Introduction (motivation, background)
- Related Work (satellite PM2.5, transfer learning)
- Data & Methods (EO features, models, transfer setup)
- Results (table, plots, performance metrics)
- Discussion (insights, limits, future work)
- Conclusion
- References + Attribution
- Appendices (optional): code summary, extra figures
