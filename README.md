# Benedict von Schubert

Physics student at Heidelberg University, working on ML for physical sensor and imaging data and on differentiable simulation.

- **AstroAI Lab, Heidelberg:** differentiable MHD simulations in JAX; my Bachelor's thesis builds ML on top of them.
- **Zap Energy (summer 2026):** experimental diagnostics team. I built the interferometry analysis pipeline that is now the team's baseline diagnostic. That code is proprietary, so it isn't here.

### Projects

- **[OpenEarth](https://github.com/Benedict-vs/OpenEarth):** methane super-emitter detection in Sentinel-2 imagery. Physics-based retrieval and flux quantification with Monte-Carlo uncertainty, plus a U-Net candidate ranker (cluster-held-out F1 0.57 vs 0.42 for the physics baseline).
- **[scriptify](https://github.com/Benedict-vs/scriptify):** turns a semester of lecture recordings into a LaTeX textbook with full derivations and redrawn figures, every block linked to its source timestamp. Scripts made with it are becoming official course material for Heidelberg lectures.
- **[bomberman_RL](https://github.com/Benedict-vs/bomberman_RL):** reinforcement-learning agents for four-player Bomberman (team of three, Machine Learning Essentials). I led the tabular Q-learning agent we submitted to the course tournament, built on a hand-built feature map with curriculum training: 4.02 points per round against three rule-based opponents, vs 3.25 for the rule-based agent in the same slot.
- **[astronomix](https://github.com/Benedict-vs/astronomix):** my fork of the lab's differentiable JAX MHD code, used for galactic-outflow simulations at up to 2048³ cells (8.6 billion) on multi-node GPU clusters. I added multi-node sharding, checkpoint/restart, radiative cooling with FUV heating (including a fused GPU kernel), and forward-mode sensitivity maps of the outflow to supernova mass and energy loading.

Python · JAX · C++ · NumPy · FastAPI · Google Earth Engine · LaTeX
