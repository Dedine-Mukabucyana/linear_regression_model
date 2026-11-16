# GBV Partner Violence Prevalence Predictor

**Mission and problem:** Gender-based violence (GBV) often goes unreported. My mission is an anonymous reporting platform where every report is followed up by the Isange One Stop Centre and the Rwanda Investigation Bureau (RIB).
Isange, RIB and the platform team have limited staff, so they need to know which groups of women face the highest partner violence to decide where to focus outreach and follow-up.
This project predicts the percentage of women in a group who experienced physical and/or sexual violence from a husband or partner in the last 12 months.

## Dataset

- **Source:** Demographic and Health Surveys (DHS) Program, downloaded with the free DHS API: https://api.dhsprogram.com (program website: https://dhsprogram.com).
- **Target:** `violence_pct`, indicator `DV_SPV1_W_PAS` (physical and/or sexual violence by a husband/partner in the last 12 months).
- **Rows and features:** 1,582 rows and 9 features (3 categorical: country, group category, group label; 6 numeric: survey year, women's literacy, secondary education, bank account access, acceptance of wife beating, fertility rate).
- **Scope:** 10 East and Central African countries (Rwanda, Burundi, Uganda, Kenya, Tanzania, DR Congo, Ethiopia, Zambia, Malawi, Mozambique). Rwanda alone has too few rows for a reliable model.
- **How it was built:** `summative/linear_regression/build_dataset.py` downloads the target broken down by country, survey year and group, and adds national context indicators from the same survey.

## Models (summative/linear_regression/multivariate.ipynb)

Four scikit-learn models were compared on the same test set by MSE: SGDRegressor, LinearRegression, DecisionTreeRegressor and RandomForestRegressor.

| Model | Test MSE |
|---|---|
| RandomForestRegressor (best, saved) | 6.032 |
| DecisionTreeRegressor | 6.324 |
| LinearRegression | 9.066 |
| SGDRegressor | 9.479 |

The best model is saved with its preprocessing as a Pipeline in `summative/API/best_model.pkl`.

## Public API (Swagger UI)

**Swagger UI:** https://gbv-prevalence-api.onrender.com/docs

Routes: `POST /predict`, `POST /retrain` (upload a CSV), `GET /options`, `GET /docs`.
The free Render service sleeps after 15 minutes without traffic, so the first request can take up to a minute.

## Video demo

**YouTube:** <PASTE-YOUR-YOUTUBE-LINK>

## Run the mobile app

1. Install Flutter and an Android emulator or connect a phone with USB debugging.
2. Clone the repository: `git clone https://github.com/Dedine-Mukabucyana/linear_regression_model.git`
3. Go to `summative/FlutterApp` and run `flutter pub get`.
4. Set the API address in the app code to the Swagger link above (without `/docs`).
5. Run `flutter run` and choose your emulator or phone.
6. Fill in the 9 fields and tap **Predict**.
