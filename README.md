# conversion-impact-synthetic-control
A project demonstrating synthetic control-based causal inference to measure conversion uplift from individual levers in e-commerce environment.
# 🧠 Causal Inference for E-commerce Conversion – An IPTW Approach

This project demonstrates how to estimate the **causal effect of business drivers** on user conversion in an e-commerce setting using **observational data**, where randomized experiments (A/B tests) are not feasible. The core methodology applied here is **Inverse Probability of Treatment Weighting (IPTW)**, a technique from the causal inference toolkit.

The analysis and code are based on real-world projects from the e-commerce domain (originally developed in a corporate environment), but are recreated using **synthetic data** for public sharing, ensuring **no proprietary information** is exposed.

---

## 🎯 Problem Statement

In digital businesses, growth teams often activate levers like:
- **Selection** (product availability, sizes, quality),
- **Pricing** (discounts, price index),
- **User-generated content** (ratings, reviews),
- **Service features** (COD availability, delivery promise).

This project evaluates:  
> _"What is the causal impact of activating these levers on conversion, when randomization isn't possible?"_

---

## 🧰 Methodology: Inverse Probability of Treatment Weighting (IPTW)

We frame the problem as a **binary treatment effect estimation**:
- **Treatment**: Whether a given user/product pair was exposed to a favorable condition (e.g., better pricing index).
- **Outcome**: Whether the user converted (made a purchase).

### ✅ Key Steps:

1. **Feature Selection**  
   - Identify confounders (pre-treatment variables influencing both treatment & outcome)
   - Select top features via a gradient-boosted tree model

2. **Propensity Score Modeling**  
   - Train a classifier to estimate the probability of treatment given covariates

3. **Weighting with IPTW**  
   - Compute weights: `1/p(t|X)` for treated, `1/(1-p(t|X))` for control
   - Weight the dataset to create a **balanced pseudo-population**

4. **Balance Check**  
   - Use Standardized Mean Difference (SMD) to ensure covariates are balanced after weighting

5. **Causal Effect Estimation**  
   - Estimate the **Average Treatment Effect (ATE)** and interpret impact on conversion

---

## 🧪 Data

The dataset used here is **synthetic**, generated to mirror the structure of real-world e-commerce logs.  
It contains:
- **User-product-level rows**
- Treatment flag
- Conversion outcome
- Features such as product rating, price index, page views, seller type, and delivery availability
---

## 🗂️ Repository Structure

```bash
causal-conversion-model/
│
├── data/
│   └── simulated_conversion_data.csv
├── notebooks/
│   └── conversion_uplift_model.ipynb  # full analysis in notebook format
├── src/
│   └── causal_utils.py                # functions for IPTW, SMD, ATE, etc.
├── README.md
└── requirements.txt


⸻

🛠️ Tech Stack
	•	Python 3.9+
	•	NumPy / Pandas
	•	scikit-learn
	•	XGBoost
	•	statsmodels
	•	matplotlib / seaborn

⸻

📈 Insights (Generalized)

The causal inference approach enables:
	•	Measuring true incremental impact of business levers on conversion
	•	De-biasing estimates that would otherwise be confounded by targeting effects
	•	Enabling data-driven investment decisions across levers (e.g., discounting vs selection)

⸻

⚠️ Limitations & Challenges
	•	Unmeasured confounding: Assumes all relevant confounders are observed
	•	Positivity: Requires overlap in treatment probability across covariate space
	•	Model sensitivity: Estimates are sensitive to how propensity scores are modeled
	•	Extreme weights: May require trimming or regularization

⸻

📜 Disclaimer

This project is a generalized and privacy-safe reconstruction of internal work originally done in a corporate setting. All data is simulated. No confidential business logic, customer data, or proprietary metrics are shared or exposed in this repository.

⸻

🙋‍♀️ Author

Built by Hardik Kshatriya — for showcasing applied causal inference skills in real-world business problems.
