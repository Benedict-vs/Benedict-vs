# Benedict von Schubert

Physics student at Heidelberg University, working on ML for physical sensor and imaging data and on differentiable simulation.

- **AstroAI Lab, Heidelberg:** differentiable MHD simulations in JAX; my Bachelor's thesis builds ML on top of them.
- **Zap Energy (summer 2026):** experimental diagnostics team. I built the interferometry analysis pipeline that is now the team's baseline diagnostic. That code is proprietary, so it isn't here.

### Projects

- **[OpenEarth](https://github.com/Benedict-vs/OpenEarth):** methane super-emitter detection in Sentinel-2 imagery. Physics-based retrieval and flux quantification with Monte-Carlo uncertainty, plus a U-Net candidate ranker (cluster-held-out F1 0.57 vs 0.42 for the physics baseline).
- **[scriptify](https://github.com/Benedict-vs/scriptify):** turns a semester of lecture recordings into a LaTeX textbook with full derivations and redrawn figures, every block linked to its source timestamp. Scripts made with it are becoming official course material for Heidelberg lectures.
- **[bomberman_RL](https://github.com/Benedict-vs/bomberman_RL):** reinforcement-learning agents for four-player Bomberman (team of three, Machine Learning Essentials). I led the tabular Q-learning agent we submitted to the course tournament: a hand-built feature map, a four-stage training curriculum, NumPy only. Over 1000 seeded rounds against three rule-based opponents it scores 4.02 points per round, vs 3.25 for the rule-based agent in the same slot. All comparisons are paired on shared seeds with bootstrap confidence intervals.
- **[astronomix](https://github.com/Benedict-vs/astronomix):** my fork of the lab's JAX MHD code, used for large galactic-outflow simulations.

Python · JAX · C++ · NumPy · FastAPI · Google Earth Engine · LaTeX
