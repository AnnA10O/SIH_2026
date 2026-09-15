# PS 26077 — AI Hyper-Local Severe Weather Nowcasting
## Research Report & Build Plan

*"Nowcasting" just means: predicting weather for the very near future — minutes to a few hours ahead — as opposed to "forecasting," which usually means a day or more ahead. It's a different, harder problem because it needs to react to storms that are already forming, right now.*

---

## 1. Where this field actually stands today (plain-language summary)

Nowcasting has moved fast in the last five years — Google, Microsoft, and NOAA all now run real AI nowcasting systems, and Google's satellite-based model is already live for millions of people on Google Search, including in data-poor regions like India. But there's an honest, well-documented gap: every one of these systems is good at ordinary rain and struggles specifically with **sudden, small, extreme storms** — exactly what a cloudburst is. India's own systems (IMD, ISRO/MOSDAC) already do some of this — cloudburst alerts exist for the Western Himalayas, and a new flood-forecasting chain (C-FLOOD) launched in 2025 — but even that system's own documentation admits cloudbursts are "sub-grid events," meaning they're smaller than what the underlying models can actually see. So this is not an empty field, and it's not a solved one either: there's a real, current, citable research gap specifically around the *extreme, hyper-local, 2-6 hour* slice of the problem, which is exactly where PS 26077 sits.

---

## 2. PART 1 — What's already built

### Global systems

| System | Who built it | Predicts | Lead time | Deployed or research? |
|---|---|---|---|---|
| **DGMR** (Deep Generative Model of Rainfall) | Google DeepMind + UK Met Office | Radar rainfall images | Up to 90 min | Extensively tested with real Met Office forecasters (they preferred it over other methods in blind tests); code released. Not a public consumer app. |
| **NowcastNet** | Tsinghua University + Google (Zhang et al., *Nature*, 2023) | Radar rainfall, with real physics baked in (see Part 5) | Up to 3 hours | Research system, evaluated with meteorologists on US and China data. Specifically better at **extreme** rain than DGMR. |
| **Global MetNet** | Google (Agrawal et al., 2025, arXiv) | Satellite-only rainfall (no radar needed — built for radar-sparse regions like India) | Up to 12 hours | **Deployed right now** — live on Google Search for millions of users worldwide. |
| **MSNowcasting** | Microsoft | Radar rainfall (ConvLSTM-based) | Short-term | Operational inside Microsoft/Bing Weather. |
| **NOAA AI weather models (AIGFS)** | NOAA | Global weather (broader than just nowcasting) | Hours to days | Deployed Dec 2025–Feb 2026. Good context, but this is more medium-range than hyper-local nowcasting. |
| **pysteps** | Open-source community (Pulkkinen et al., 2019) | Radar rainfall via optical flow (tracks rain patterns and extrapolates their movement — no AI needed) | Up to ~1 hour reliably | The "old guard" baseline many national weather agencies already run operationally. Free, open-source — see Part 3. |

### India specifically

- **IMD runs 39 Doppler Weather Radars (DWRs)** today, expanding to 70 by 2025-26 under a named government programme called **Mission Mausam** — which explicitly aims to "leverage AI for better hyperlocal forecasts." This is genuinely the same goal as PS 26077, happening at government scale right now.
- **ISRO's MOSDAC portal already offers cloud-burst and heavy-rain nowcasting for the Western Himalayan region**, using satellite data. It exists — but it's regionally limited (just the Western Himalayas) and is not framed as a full village/block-level 2–6 hour AI system.
- **C-FLOOD**, India's Unified Inundation Forecasting System, launched in 2025 (Ministry of Jal Shakti). It chains IMD's weather models → hydrological models → flood-extent maps → district alerts. Its own documentation states plainly: **"cloudbursts are sub-grid events for most NWP models"** — i.e., even this brand-new system admits it can't reliably catch a cloudburst because the storm is physically smaller than the grid boxes the model "sees" in.
- **A well-known, openly acknowledged blind spot:** if a cloudburst happens a few kilometres from the nearest radar or weather station — common in the Himalayas, where mountains block radar signal and stations are sparse — it often isn't recorded at all.
- IMD already has city flood-warning systems for Mumbai and Chennai (since ~2020), with Bengaluru and Kolkata versions in progress. Those are urban drainage-flood systems though — a different problem from mountain cloudburst prediction.

**Bottom line: you are not building this from zero, and you are not competing with nothing.** You're building the *specific missing piece* — sub-grid-scale, 2–6 hour, hyper-local — that India's own newest systems openly admit they don't yet cover well.

---

## 3. PART 2 — What's still not solved (the honest gap)

### Why is a 2-6 hour cloudburst so much harder than tomorrow's rain forecast?

- **Cloudbursts are tiny.** IMD defines one as 100mm+ of rain in an hour over just 20-30 sq km. Most weather models carve the atmosphere into grid boxes bigger than that — so the storm can literally be smaller than a single "pixel" the model can think in. This is called being a **"sub-grid" event**.
- **They form and explode fast** — often in under an hour, from calm to extreme. Traditional physics-based weather models solve huge systems of equations that take real computer time to run; by the time the model catches up, the storm may already be over.
- **In India specifically, the data is patchy exactly where the risk is highest** — Himalayan and Western Ghats terrain blocks radar line-of-sight and has fewer weather stations, so the AI has less to learn from in the hardest, highest-stakes places.

### Real papers that openly say this is still unsolved

| Paper | Authors / Venue / Year | What it says, simply |
|---|---|---|
| *Skilful nowcasting of extreme precipitation with NowcastNet* | Zhang et al., **Nature**, 2023 | States plainly, in its own opening lines, that physics-based models "struggle to capture... convective initiation" (the moment a storm starts) and that pure data-driven AI models "fail to obey intrinsic physical laws" — i.e., both major approaches have a real, different weakness. |
| *Brief communication: Training of AI-based nowcasting models for rainfall early warning should take into account user requirements* | Heistermann et al., **Natural Hazards and Earth System Sciences**, 2025 | Says it "appears to remain difficult to successfully learn precipitation dynamics over a wide range of weather conditions" — a direct, recent, open admission that today's AI models don't generalise well across different storm types. |
| *Skilful precipitation nowcasting using deep generative models of radar* (DGMR) | Ravuri et al., **Nature**, 2021 | The model that impressed real forecasters — but by its own results, it specifically struggled with the most **extreme** precipitation events, which is exactly the cloudburst case. |

### What the best systems are still bad at

- **False alarms and missed small storms** — a storm smaller than the model's "vision" can be entirely missed, or a real one over-predicted.
- **Blurry, "averaged-out" predictions the further ahead you look** — RainNet's own paper notes that beyond about 5 minutes, its predictions get progressively smoother, and sharp, dangerous rain intensities get smoothed away — exactly the detail you most need for a warning.
- **Needing supercomputers** — physics-based models like NOAA's HRRR need serious computing power, which a student team (and even many national weather services) simply doesn't have.
- **Falling apart in radar-sparse regions** — this is explicitly why Google built Global MetNet as a satellite-only alternative — most AI nowcasting research assumes dense radar coverage that most of India (outside the plains) doesn't have.

---

## 4. PART 3 — Data and tools actually available to you

### Free and public (usable right now, no special access needed)

| Source | What it gives you | Notes |
|---|---|---|
| **MOSDAC** (mosdac.gov.in) — ISRO's own data portal | INSAT-3D/3DR/3DS satellite imagery, radar mosaics, ground station (AWS) data, existing cloud-burst/heavy-rain alert products | Free ("freeware" license), needs a simple registration. This is your single best India-specific data source. |
| **NASA GPM (Global Precipitation Measurement)**, specifically the **IMERG** product | Global satellite rainfall estimates, updated every 30 minutes, covers all of India | Free, no registration friction, very widely used in exactly this kind of research. |
| **ERA5** (ECMWF reanalysis) | Historical weather data, good for training and background context | Free, but not real-time — use it for training data, not live prediction. |
| **data.gov.in** | Some government rainfall/hydromet datasets | Coverage varies — check what's currently listed for your target region. |
| **Central Water Commission (CWC)** | 332 flood-forecast stations, 700+ river-level stations | Some data publicly shared, useful if your area has a river/flood angle. |
| **ISRO Bhuvan / CartoDEM** | Digital elevation data (terrain height), 30m resolution | Useful for hydrology — cloudbursts trigger flash floods that follow terrain. |

### Needs special government access (don't plan around these)

- **Raw, full-resolution IMD Doppler Weather Radar data** — unlike the US (where NEXRAD radar data is genuinely open), IMD does not publicly release live raw radar sweeps. You may see processed radar images on the Mausam website/app, but not the underlying data feed.
- **IMD's dense AWS network live feed** — not openly API-accessible to the public.
- **NCMRWF's operational model output** — generally research/institutional access only.

### Open-source code and pretrained models — don't start from zero

- **pysteps** (`github.com/pySTEPS/pysteps`) — the standard open-source nowcasting toolkit. Well documented, has a ready-to-run Google Colab example, and includes sample radar data. **Get this running in week 1** — it's your safety net.
- **RainNet** (`github.com/hydrogo/rainnet`) — a convolutional neural network for radar nowcasting, with pretrained weights already available.
- **pysteps-dgmr-nowcasts plugin** — lets you run DeepMind's actual DGMR model inside pysteps; it auto-downloads the pretrained weights for you.
- **MetNet reimplementation** (`github.com/openclimatefix/metnet`) — an open-source community rebuild of Google's MetNet, maintained by OpenClimateFix (a nonprofit focused on open climate AI).
- **Weather4cast** (NeurIPS challenge) — a public satellite-nowcasting benchmark dataset with baseline code, useful to compare yourself against a known standard.

---

## 5. PART 4 — A realistic 4-6 week project

### What to cut, and why

- **Cut: covering all of India.** Pick **one** cloudburst-prone region (e.g., a specific district in Uttarakhand, Himachal, or the Western Ghats). Terrain-specific tuning is far more tractable, and "we solved it for this real place" is a stronger, more honest demo than a shallow attempt at the whole country.
- **Cut: building a physics-informed model from scratch like NowcastNet.** That's a multi-year research effort. Instead: **start from pysteps** (working in days, not weeks), then add a deep-learning correction layer on top, trained on your region's historical data.
- **Cut: promising reliable performance across the full 2-6 hour window.** Even NowcastNet — a *Nature* paper from a team with far more resources than you — caps out around 3 hours. Aim to nail **0-2 hours really well**, and present 2-6 hours as an honest "experimental extension," not a guaranteed capability.

### What gives the most "wow" for the least building effort

Combine free satellite data (covers the whole country, including sensor-sparse hills) with the pysteps baseline and a simple trained correction model — then **wire the output into a real WhatsApp/SMS alert**. In our experience with judges, a phone actually buzzing with a real warning message in the local language during your live demo creates far more impact than a slightly better accuracy number ever will.

---

## 6. PART 5 — Clever engineering tricks that genuinely fit this problem

1. **Cheap check first, expensive model only when needed (compute-gating).** Run a fast, lightweight check across your whole region continuously — for example, watching how fast satellite cloud-top temperature is dropping (a proxy for a storm rapidly building upward). Only fire up your full, expensive nowcasting model for the specific small area that looks unusual. This is genuinely how a country-scale alert system has to work, and it lets your "hyper-local" system actually run continuously on modest hardware.

2. **Bake real physics into the model, not just data (physics-informed AI).** NowcastNet's key trick, explained simply: rain doesn't teleport or vanish — the total amount of moisture moving through the air has to be conserved and carried by wind (this is called the "continuity equation," a basic physics law). By building that rule directly into the model instead of hoping the AI discovers it from data alone, NowcastNet needs less training data and — importantly for your pitch — gives a judge a real, physically grounded answer to "why does your model do this," instead of "the neural network decided so."

3. **Fuse cheap ground sensors with satellite data to fix a known bias (sensor-fusion cost-hack).** Satellite rainfall estimates are well known to be systematically off in mountainous terrain. Your region's handful of real ground rain-gauge/AWS readings can be used to locally correct the satellite estimate for the whole surrounding area — a cheap sensor calibrating an otherwise-imprecise expensive one.

4. **Predict "will it cross a dangerous threshold?" instead of the exact rainfall number.** Straight from the 2025 NHESS paper (Part 2): its authors' own hypothesis is that models struggle because they're asked to predict *exact* rainfall everywhere, across every kind of weather. Reframing the task as "will rainfall exceed [dangerous threshold] in this area, yes/no/probability" is a simpler, more learnable question — and it's directly the question a warning system actually needs answered.

---

## 7. PART 6 — Full plan for using Spiking Neural Networks (SNN)

### 6.1 Why SNN here, in simple words

**What is an SNN, really?** A normal neural network (CNN/LSTM) looks at a fixed slice of data — a frame, a time-step — and does a full round of math on it, every single step, using continuous numbers. It's always "on." An SNN is modeled on how a real brain neuron works: it sits essentially silent, using almost no energy, quietly accumulating input — and only when enough builds up does it suddenly fire a single sharp pulse (a "spike"), then go quiet again. So the core difference is: **always-on continuous math vs. mostly-silent, event-triggered pulses.**

**What real research exists for weather/radar specifically?**

| Paper | Authors / Year | What it actually did | How close is it to PS 26077? |
|---|---|---|---|
| *Enhanced smart weather prediction... using Binarized Spiking Neural Networks* | Amanullah, Ananthajothi & Divya, 2025 (*Knowledge and Information Systems*) | A genuine SNN + weather paper — but it does simple **rain / no-rain classification from tabular weather-station numbers**, not spatial radar/satellite image prediction. | Distant. Same broad topic, much simpler task than what you need. |
| *QLIF-CAST* | Marchisio, Ebrahim, Innan, Kashif, Shafique, 2026 (arXiv) | A quantum-enhanced spiking neuron tested on short-term multivariate weather and air-quality time series; beat a matched classical spiking baseline by ~15% lower error. | Closer (genuinely weather time-series), but adds a whole extra layer of quantum-computing complexity you don't need. |
| *NeuroRadar: A Neuromorphic Radar Sensor for Low-Power IoT Systems* | Zheng et al., **SenSys**, 2023 | Not weather — but a real, working event-driven/spiking approach to processing *radar* signals on low-power IoT hardware. | The closest real-world precedent for "SNN + radar-like signal + power-constrained edge device." |
| *EdgeSpike* | 2026 (arXiv) | Tests SNNs on five edge-sensing tasks including 77GHz radar activity classification, on real low-power hardware. | Same pattern — SNN genuinely works for edge/radar-adjacent sensing, not for big spatial reasoning. |

**Honest verdict:** No one has shown an SNN successfully doing the actual job PS 26077 needs — reading a 2D radar/satellite image sequence and predicting where a storm goes. Every serious high-accuracy system in Part 1 (DGMR, NowcastNet, Global MetNet) uses ordinary deep learning for that job, not SNNs. Betting your core forecasting accuracy on an unproven SNN approach would be a real risk. **SNN is a genuinely good, defensible fit for a specific, narrower slice of the pipeline — not the main forecasting brain.**

### 6.2 Where exactly SNN belongs in the pipeline

| Stage | Best fit | Why |
|---|---|---|
| 1. Raw data collection (sensors/AWS/radar/satellite) | **SNN — specifically at the ground-station/edge level** | A remote, battery/solar-powered hillside rain gauge doesn't need to constantly stream data. An event-driven "wake up and send data only when the reading spikes unusually fast" logic is exactly what SNNs are built for — and it directly matches your rubric's "power-constrained, event-driven" scenario. |
| 2. Data cleaning | Normal code | Nothing SNN-specific to gain here. |
| 3. Main prediction model | **Normal deep learning** (CNN/ConvLSTM, or pysteps + a trained correction layer) | This is the big spatial-reasoning job — combining data across a wide area at once. Every real high-performing system uses this kind of architecture. This is where your accuracy comes from — don't gamble it. |
| 4. Turning prediction into alert | Normal code/rules | No SNN benefit. |

**Direct answer to the trade-off question:** yes — SNN belongs on the ground-station side, not the cloud-side forecasting model. The cloud side needs to reason over large-area spatial data all at once, which suits a model built for exactly that (CNN/ConvLSTM/transformer). The ground-station side needs to sit quietly on a tiny power budget for months and only act when something unusual happens — which is exactly what a spiking, event-driven design is for.

### 6.3 Tech stack recommendation

- **SNN framework: snnTorch.** Confirmed as the most widely used SNN library, built directly on PyTorch, with the most beginner tutorials of any SNN library — good for learning in days, not weeks. **Norse** is a solid second choice (also PyTorch-based, cleaner functional style, smaller community). **Lava** is Intel's official framework, but it's specifically built to target real Intel Loihi chips — skip it unless you somehow get actual Loihi hardware. **BindsNET**, **Brian2**, and **Nengo** lean more toward neuroscience research and reinforcement learning than "ship a working demo" — skip them here.
- **Main forecasting model: PyTorch** (not TensorFlow) — since snnTorch/Norse are PyTorch-native, keeping everything in one ecosystem avoids painful format conversion between two different deep-learning stacks.
- **Data processing:** Python's `xarray` + `netCDF4`/`cfgrib` libraries — satellite and radar data usually comes in **NetCDF** or **GRIB** format (just standardized "science data file" formats used across weather/climate data); these libraries read them straight into normal numeric arrays.
- **Backend/API:** **FastAPI** (Python) — lightweight, quick for a small team to wrap the model in a simple "send a location, get a forecast" web service.
- **Frontend/alerts:** **Leaflet** or **Mapbox** (free tiers) for the map visualization. For alert delivery, **Twilio's WhatsApp/SMS API** (has a free trial tier) is the standard, easy way to send a real phone alert live during a demo.
- **Hardware:** everything above runs fine on a normal laptop or a free cloud GPU (Google Colab, Kaggle). **You do not need real neuromorphic chips** (Intel Loihi 2, BrainChip Akida) — snnTorch/Norse simulate genuine spiking behavior in ordinary software on an ordinary GPU. You get the real event-driven logic and can estimate the theoretical energy savings on paper; what you don't get is the *actual* hardware power-draw benefit, since that only shows up on real neuromorphic silicon. Be upfront about that distinction with judges (see 6.5).

### 6.4 Week-by-week build plan (4-6 weeks)

**Week 1 — Set up and learn**
- Get **pysteps** running end-to-end on sample public radar/satellite data. This gives you a working, if basic, nowcast pipeline almost immediately — your safety net from day one.
- Pull real data for your one chosen region from MOSDAC and NASA GPM.
- Whoever's doing the SNN piece works through snnTorch's official tutorials (a few hours) and gets a toy SNN running on a simple public dataset — confirming the tooling works before touching real weather data.

**Weeks 2-3 — Build the riskiest parts first**
- *Riskiest piece #1 (main model):* start training a deep-learning correction layer on top of pysteps for your specific region immediately — this is your core deliverable and the most likely to eat unexpected time.
- *Riskiest piece #2 (SNN):* build a small SNN "event detector." Feed it a simulated stream of rain-gauge/AWS readings (your historical data, replayed as if live) and train it to fire a "send data now" spike only when the pattern looks like a storm building — and stay quiet on ordinary weather. Keep this scoped small — it's a helper component, not your main model.
- Test both pieces separately before combining them.

**Week 4 — Integration**
- Wire the SNN "when to send" trigger into the ingestion side of your pipeline, and the main deep-learning model to produce the 0-2 hour prediction (stretch goal: extend toward 6 hours, clearly labeled as experimental).
- Build the map visualization and connect the WhatsApp/SMS alert API.

**Weeks 5-6 — Polish, demo prep, fallback plan**
- **If the SNN piece isn't behaving well in time:** it's a secondary component by design. Demo the full system with a simple threshold rule standing in for the SNN's job, and separately show your trained SNN's own validation results (accuracy/efficiency numbers) as a side result — you still get full credit for the engineering depth without it gating your live demo.
- **Minimum working demo, if time runs out completely:** a pysteps baseline nowcast for your one region, shown on a map, with one real historical cloudburst event replayed to show "here's what our alert would have said, and how many minutes before the real event it would have fired." That alone is a complete, honest, demonstrable product.

### 6.5 How to explain the SNN choice to judges

> "We use a normal deep learning model for the actual storm prediction, because that's genuinely what the best systems in the world use, and we didn't want to gamble our accuracy on something unproven. But every remote hillside sensor in cloudburst-prone terrain runs on a small battery, and constantly streaming data drains it fast. So we used a spiking neural network — a brain-inspired model that stays silent and uses almost no power until it detects a sudden, storm-like change — to decide, right at the sensor, when data is actually worth sending. It's the right tool for the power-constrained edge, not for the big spatial reasoning job."

---

## 8. PART 7 — The mathematical model

This is the actual math behind the pipeline in Parts 4-6, laid out stage by stage. Each equation is tied to a real paper already cited above, so you can point a judge to the source. **You don't need to implement all of it** — the "core" pieces (7.1 and 7.2) are enough for a working 4-6 week demo; 7.3 and 7.4 are the "extra technical depth" pieces to add if time allows.

**The pipeline, in one line:** raw radar/satellite frames → (7.1) simple physics-based motion extrapolation → (7.2) a neural network learns the "correction" the simple physics misses → (7.3) a physics rule keeps that correction honest → (7.4) turn the prediction into an exceedance probability → (7.5) SNN decides when a ground sensor even sends data → (7.6) alert fires.

### 7.1 The baseline: rain doesn't teleport (the advection equation)

The simplest honest model of rain movement is: a patch of rain gets carried by the wind, without changing shape or intensity. In physics, this "no source, no sink, just carried along" rule is called the **advection equation** (also called the continuity equation with no source term):

$$\frac{\partial X}{\partial t} + \mathbf{v} \cdot \nabla X = 0$$

*In plain words:* $X(s,t)$ is the rain intensity (mm/hr) at location $s$ and time $t$. This equation just says "the only reason rain intensity at a point changes is that the wind $\mathbf{v}$ is physically carrying the rain pattern past that point" — no storm is allowed to grow, shrink, or appear from nowhere under this equation alone.

**Estimating the wind field $\mathbf{v}$** from two consecutive radar/satellite frames is done with **optical flow** — the same math used to estimate motion in video. The classic Lucas-Kanade method solves, for each small local patch of the image:

$$\min_{\mathbf{v}} \sum_{\text{patch}} \left( \nabla X \cdot \mathbf{v} + \frac{\partial X}{\partial t} \right)^2$$

*In plain words:* find the motion vector $\mathbf{v}$ that best explains how this patch of the image changed between two frames — this is exactly what pysteps computes internally.

Once you have $\mathbf{v}$, the forecast is just "trace backward along the wind and copy that value forward" (a **semi-Lagrangian** extrapolation, pysteps' core method):

$$\hat{X}(s, t_0+\tau) = X(s - \tau\mathbf{v}(s,t_0),\ t_0)$$

*In plain words:* "to predict the rain at this spot in $\tau$ hours, look at where that rain parcel is coming from right now, and assume it just moves here unchanged." This alone is your Week 1 safety-net baseline.

### 7.2 The correction: a neural network learns what pure physics misses

Pure advection can't capture a storm **intensifying, weakening, or forming from nothing** — real convective storms do all three. So you add a learned correction term on top:

$$\hat{X}_{\text{final}}(s, t_0+\tau) = \hat{X}(s, t_0+\tau) + G_\theta\big(X(t_0-T{:}t_0),\ \mathbf{v},\ \tau\big)$$

*In plain words:* final prediction = simple physics guess + "whatever the physics guess got wrong," learned by a neural network $G_\theta$ from historical data.

$G_\theta$ is a **ConvLSTM** — the standard building block for this exact task (Shi et al., *NeurIPS*, 2015; the same family of model used in RainNet, MetNet, and MSNowcasting from Part 1). It's an LSTM (a network with memory across time) where every multiplication is replaced by a convolution (so it has memory across time *and* understands spatial neighborhoods):

$$
\begin{aligned}
i_t &= \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + b_f) \\
o_t &= \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + b_o) \\
g_t &= \tanh(W_{xg} * X_t + W_{hg} * H_{t-1} + b_g) \\
C_t &= f_t \odot C_{t-1} + i_t \odot g_t \\
H_t &= o_t \odot \tanh(C_t)
\end{aligned}
$$

*In plain words:* at each time step, the network decides what to "forget" ($f_t$), what new information to "remember" ($i_t$, $g_t$), and what to actually output ($o_t$) — all as full 2D maps via convolution ($*$), so it's learning spatial patterns of storm growth/decay, not just a single number. $\odot$ just means "multiply element by element."

**The loss function matters more than the architecture here.** Plain mean-squared-error (MSE) causes exactly the "blurry, averaged-out prediction" problem flagged in Part 2 (RainNet's own finding) — because MSE is minimized by predicting something safely close to average everywhere. Fix it with a **weighted loss** that penalizes missing heavy rain much more than missing light rain:

$$\mathcal{L}_{\text{data}} = \frac{1}{N}\sum_{i=1}^{N} w(X_i)\cdot (\hat{X}_i - X_i)^2, \qquad w(X) = 1 + X$$

*In plain words:* getting a 100mm/hr cloudburst wrong costs the model far more (in training) than getting a light drizzle wrong — which pushes the network to actually try to predict extremes instead of playing it safe.

### 7.3 The physics check: NowcastNet's trick, in equation form

NowcastNet's key idea (Zhang et al., *Nature*, 2023 — the paper that specifically beats other models on **extreme** rain) is to penalize the network whenever its prediction breaks the continuity equation from 7.1:

$$\mathcal{L}_{\text{physics}} = \frac{1}{N}\sum \left| \frac{\partial \hat{X}}{\partial t} + \nabla\cdot(\hat{X}\,\mathbf{v}) \right|^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda\, \mathcal{L}_{\text{physics}}$$

*In plain words:* on top of "match the real data" ($\mathcal{L}_{\text{data}}$), add "don't violate the physical law that rain-mass is conserved" ($\mathcal{L}_{\text{physics}}$), weighted by a tunable knob $\lambda$. This is what lets the network learn from *less* data — it's not guessing the physics from scratch, only the part physics doesn't already explain.

### 7.4 Turning a number into a warning: exceedance probability

A single predicted number hides risk — you actually want "how likely is this to cross the cloudburst threshold?" A simple, standard way (used by DGMR-style ensemble models, Part 1) is to generate $N$ slightly different plausible forecasts (by sampling random noise $z$ into the model) and count how many cross the danger line:

$$\hat{P}\big(X(s,t_0+\tau) \geq \theta_{\text{danger}}\big) = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[\hat{X}_i(s,\tau) \geq \theta_{\text{danger}}\big]$$

*In plain words:* run your model $N$ times with slightly different random seeds, and your exceedance probability is just "what fraction of those runs predicted a cloudburst." $\theta_{\text{danger}}$ = IMD's own definition, 100mm/hr. $\mathbb{1}[\cdot]$ just means "1 if true, 0 if false."

**Fire the alert when:**

$$\text{Alert}(s) = 1 \iff \hat{P}(\text{exceed}) \geq p_{\text{threshold}} \ \text{ for some } \tau \in [2h, 6h]$$

*In plain words:* $p_{\text{threshold}}$ is a dial you tune using real historical events — turn it down and you catch more real storms but cry wolf more often; turn it up and you're quieter but risk missing one. Pick it using a precision-recall curve on your validation events, not a guess.

### 7.5 The SNN sensor trigger: Leaky Integrate-and-Fire, in equation form

This is the actual math snnTorch runs under the hood for the ground-station piece from Part 6. The **Leaky Integrate-and-Fire (LIF)** neuron:

$$V[t] = \beta\, V[t-1] + I[t] - \theta\, S[t-1]$$

$$S[t] = 1 \ \text{ if } V[t] \geq \theta, \text{ else } 0$$

*In plain words:* $V[t]$ is like a bucket filling with water ($I[t]$, the incoming signal — e.g. how fast the rain-gauge reading is rising), $\beta \in (0,1)$ is a small leak in the bucket (so slow, gradual changes don't accidentally trigger anything — only a genuinely sharp spike fills it fast enough to matter), and once the bucket crosses the rim $\theta$, it fires a spike ($S[t]=1$, "send the data now") and empties.

Since $S[t]$ is a hard on/off step, it technically can't be trained with normal backpropagation (its slope is zero almost everywhere). snnTorch fixes this during training only, with a smooth stand-in (a **surrogate gradient**, e.g. the fast-sigmoid):

$$\frac{\partial S}{\partial V} \approx \frac{1}{(1+k|V-\theta|)^2}$$

*In plain words:* during training, pretend the on/off switch is actually a smooth dial, so the network can still learn "how close was I to firing" — at actual run-time, it's back to a real hard spike.

### 7.6 How you'll know if it's actually working: the standard nowcasting scorecard

Don't invent your own metric — use the same ones DGMR/NowcastNet/pysteps papers report, so your numbers are comparable to published work. At a given rain-rate threshold (e.g. 1, 10, 50 mm/hr), count:

- **Hits** = you predicted rain above the threshold, and it really happened
- **Misses** = it happened, you didn't predict it
- **False alarms** = you predicted it, it didn't happen

$$\text{CSI} = \frac{\text{Hits}}{\text{Hits}+\text{Misses}+\text{False Alarms}}, \quad \text{POD} = \frac{\text{Hits}}{\text{Hits}+\text{Misses}}, \quad \text{FAR} = \frac{\text{False Alarms}}{\text{Hits}+\text{False Alarms}}$$

*In plain words:* **CSI** (Critical Success Index) is the single number every nowcasting paper leads with — closer to 1 is better, and it punishes both missing storms and crying wolf. **POD** tells you "of the real storms, how many did we catch." **FAR** tells you "of our warnings, how many were false alarms." Report CSI at a low threshold (1mm/hr, easy) and a high one (50mm/hr, the actual cloudburst-relevant number) separately — a judge who knows this field will specifically ask for the high-threshold number, since that's where every model in Part 2 is known to struggle.

---

*Sources are cited by author/venue/year throughout so you can look up and verify each one directly — that's exactly what a judge who knows this field will ask you to do.*
