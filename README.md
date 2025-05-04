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

Then, merge images with 



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
