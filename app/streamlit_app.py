import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered",
)

# ── Minimalist custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global overrides ── */
/* FIX: Removed [class*="st-"] to prevent breaking Streamlit's Material Icons */
html, body {
    font-family: 'Inter', sans-serif;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero-emoji {
    font-size: 3.2rem;
    line-height: 1;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin: 0;
}
.hero p {
    color: #888;
    font-size: 0.92rem;
    margin-top: 0.35rem;
}

/* ── Section heading ── */
.section-heading {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.6px;
    color: #aaa;
    padding-left: 0.75rem;
    border-left: 3px solid #00d4ff;
    margin: 1.4rem 0 0.8rem;
}

/* ── Result ── */
.result-card {
    text-align: center;
    padding: 2rem 1.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(0,224,255,0.06), rgba(123,47,247,0.06));
    border: 1px solid rgba(0,224,255,0.18);
    margin: 1rem 0 1.5rem;
}
.result-card .label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #777;
    margin-bottom: 0.4rem;
}
.result-card .price {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.result-card .note {
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.7rem;
}

/* ── Tiny footer ── */
.foot {
    text-align: center;
    font-size: 0.7rem;
    color: #555;
    padding: 1.5rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "..", "models")

    model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
    te    = joblib.load(os.path.join(models_dir, "target_encoder.pkl"))

    with open(os.path.join(models_dir, "column_order.json")) as f:
        col_order = json.load(f)
    with open(os.path.join(models_dir, "metadata.json")) as f:
        meta = json.load(f)

    return model, te, col_order, meta

best_model, target_encoder, column_order, metadata = load_artifacts()

PREMIUM_BRANDS = {"audi", "bmw", "merc"}


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-emoji">🚗</div>
    <h1>Used Car Price Prediction</h1>
    <p>Get an instant AI-powered price estimate for any used car</p>
</div>
""", unsafe_allow_html=True)


# ── Metadata ─────────────────────────────────────────────────────────────────
brands = sorted(metadata.get("brands", []))
transmissions = sorted(metadata.get("transmissions", []))
fuel_types = sorted(metadata.get("fuel_types", []))
models_by_brand = metadata.get("models_by_brand", {})
year_range = metadata.get("year_range", {"min": 1995, "max": 2026})


# ── Section 1 : Vehicle ──────────────────────────────────────────────────────
st.markdown('<div class="section-heading">🔑 Vehicle</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    brand = st.selectbox("Brand", brands, format_func=str.capitalize)
with c2:
    car_model = st.selectbox("Model", sorted(models_by_brand.get(brand, [])))

c3, c4 = st.columns(2)
with c3:
    year = st.number_input("Year", min_value=year_range["min"],
                           max_value=year_range["max"], value=2018, step=1)
with c4:
    transmission = st.selectbox("Transmission", transmissions)

st.markdown('---', unsafe_allow_html=False)


# ── Section 2 : Specs ────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">⚙️ Specs</div>', unsafe_allow_html=True)

c5, c6 = st.columns(2)
with c5:
    fuel_type = st.selectbox("Fuel Type", fuel_types)
with c6:
    mileage = st.number_input("Mileage (miles)", min_value=0, value=15000, step=500)

c7, c8, c9 = st.columns(3)
with c7:
    engine_size = st.number_input("Engine Size (L)", min_value=0.0,
                                  value=1.4, step=0.1, format="%.1f")
with c8:
    tax = st.number_input("Road Tax (£)", min_value=0.0,
                          value=145.0, step=5.0, format="%.0f")
with c9:
    mpg_val = st.number_input("MPG", min_value=0.0,
                              value=55.4, step=0.5, format="%.1f")


# ── Predict ──────────────────────────────────────────────────────────────────
if st.button("🏁  Estimate Price", use_container_width=True, type="primary"):
    try:
        # ── Derived features (same as notebook) ──────────────────────
        current_year = datetime.now().year
        car_age = max(current_year - int(year), 1)
        mileage_per_year = float(mileage) / car_age
        engine_tax_ratio = (float(engine_size) / float(tax)
                            if float(tax) != 0 else 0.0)
        is_premium = 1 if brand in PREMIUM_BRANDS else 0

        # ── Target-encode the model name ─────────────────────────────
        model_enc_df = pd.DataFrame({"model": [car_model]})
        model_encoded = target_encoder.transform(model_enc_df)["model"].values[0]

        # ── Build the feature row ────────────────────────────────────
        row = {
            "model": model_encoded,
            "mileage": float(mileage),
            "mpg": float(mpg_val),
            "engineSize": float(engine_size),
            "tax": float(tax),
            "car_age": car_age,
            "mileage_per_year": mileage_per_year,
            "engine_tax_ratio": engine_tax_ratio,
            "is_premium": is_premium,
        }

        # ── One-hot encode via pd.get_dummies (mirrors notebook) ─────
        cat_mapping = {
            "transmission": transmission.lower().replace(" ", ""),
            "fuelType": fuel_type.lower(),
            "brand": brand.lower(),
        }
        for col in column_order:
            if col not in row:
                matched = False
                for prefix, value in cat_mapping.items():
                    if col.startswith(prefix + "_"):
                        suffix = col[len(prefix) + 1:]
                        row[col] = 1.0 if suffix == value else 0.0
                        matched = True
                        break
                if not matched:
                    row[col] = 0.0

        # ── Assemble DataFrame in exact training column order ────────
        input_df = pd.DataFrame([row])[column_order]

        # ── Predict ──────────────────────────────────────────────────
        prediction = best_model.predict(input_df)[0]
        price_str = f"£{prediction:,.0f}"

        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated Market Value</div>
            <div class="price">{price_str}</div>
            <div class="note">Random Forest · R² ≈ 0.96 · trained on 99 000+ listings</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📊 See computed features"):
            st.dataframe(
                input_df.T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown('<div class="foot">Car Price Prediction ML System · 2026</div>',
            unsafe_allow_html=True)