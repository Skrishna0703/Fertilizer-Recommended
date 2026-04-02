"""
Fertilizer Recommendation System - Complete Dashboard
======================================================
A production-ready Streamlit dashboard for soil analysis and 
personalized fertilizer recommendations with rich visualization.

Features:
- Two-column responsive layout
- ML-powered predictions
- Rich fertilizer information display
- NPK visualization and deficiency detection
- Custom styling with rounded cards
- Dynamic content based on predictions

Author: Fertilizer AI System
Date: 2026
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

custom_css = """
<style>
/* Black background theme */
* {
    --primary-green: #10b981;
    --primary-orange: #FF9500;
    --dark-bg: #000000;
    --card-bg: #1a1a1a;
    --white-text: #ffffff;
    --gray-text: #aaaaaa;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
    background-color: var(--dark-bg) !important;
}

p, span, div, label, li, h1, h2, h3, h4, h5, h6 {
    color: var(--white-text) !important;
}

/* Card styling */
.card {
    background-color: var(--card-bg);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #333333;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.input-card {
    background-color: var(--card-bg);
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #333333;
}

/* Result card */
.result-card {
    background-color: var(--card-bg);
    border-radius: 16px;
    padding: 2.5rem;
    border: 1px solid #333333;
    min-height: 500px;
}

/* Title styling */
.main-title {
    color: #00a8d8 !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin-bottom: 0.5rem !important;
}

.subtitle {
    color: var(--white-text) !important;
    font-size: 1.05rem !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    opacity: 0.9;
}

/* Input labels */
.input-label {
    color: var(--white-text) !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
    display: block !important;
}

.helper-text {
    color: var(--gray-text) !important;
    font-size: 0.8rem !important;
    margin-top: 0.3rem !important;
    margin-bottom: 1rem !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input {
    background-color: #1a1a1a !important;
    border: 2px solid #333333 !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
    color: var(--white-text) !important;
    font-size: 1rem !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: #00a8d8 !important;
    box-shadow: 0 0 0 3px rgba(0, 168, 216, 0.2) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] select {
    background-color: #1a1a1a !important;
    border: 2px solid #333333 !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
    color: var(--white-text) !important;
}

[data-testid="stSelectbox"] select:focus {
    border-color: #00a8d8 !important;
    box-shadow: 0 0 0 3px rgba(0, 168, 216, 0.2) !important;
}

/* Button styling */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #FF9500 0%, #FF7F00 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.9rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 4px 12px rgba(255, 149, 0, 0.3) !important;
    margin-top: 1.5rem !important;
}

[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #FF7F00 0%, #E67E00 100%) !important;
    box-shadow: 0 6px 16px rgba(255, 149, 0, 0.4) !important;
}

/* Success box */
.success-box {
    background-color: #1a3a2a !important;
    border: 2px solid #10b981 !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    color: #6ee7b7 !important;
}

/* Fertilizer name display */
.fertilizer-name {
    color: #10b981 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    text-align: center !important;
    margin: 1rem 0 !important;
}

/* Info section */
.info-section {
    background-color: transparent;
    border-left: 4px solid #FF9500;
    padding: 1rem 1rem 1rem 1.5rem;
    margin: 1rem 0;
}

/* Hide labels */
[data-testid="stNumberInput"] > label,
[data-testid="stSelectbox"] > label {
    display: none !important;
}
</style>
"""

# ============================================================================
# FERTILIZER INFORMATION DATABASE
# ============================================================================

FERTILIZER_DATABASE = {
    "Urea": {
        "name": "Urea",
        "commercial_name": "Also known as: Carbamide, NH4COONH2",
        "nutrient_composition": "Nitrogen (N): 46%, Hydrogen: 5.2%, Carbon: 20%, Oxygen: 26.6%",
        "description": "Urea is a high nitrogen-rich fertilizer containing 46% nitrogen. It is the most widely used nitrogen source in agriculture worldwide.",
        "what_it_is": "Urea is an organic compound synthetically produced from ammonia and CO2. It's highly soluble in water and easily absorbed by plants.",
        "benefits": [
            "Promotes vigorous leaf growth and greenery",
            "Improves plant color and enhances green coloration",
            "Enhances protein synthesis in plants",
            "Increases crop yield and productivity",
            "Ideal for crops needing high nitrogen",
            "Cost-effective nitrogen source"
        ],
        "deficiency_symptoms": "Stunted growth, yellowing of leaves (chlorosis), reduced yield, poor plant vigor",
        "ideal_crops": ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Vegetables", "Fruits"],
        "application_timing": "Early growth stage, vegetative stage for best results",
        "usage": "Apply 200-300 kg/hectare. Best applied during early growth stages. Split applications recommended: 40% at sowing, 60% during growth.",
        "dosage": {
            "Low N soil": "250-300 kg/ha",
            "Medium N soil": "200-250 kg/ha",
            "High N soil": "150-200 kg/ha"
        },
        "method": "Mix with soil or apply through irrigation",
        "storage": "Keep in cool, dry place. Hygroscopic - absorbs moisture easily",
        "types": [
            "Prilled Urea (granule form, 1-4mm)",
            "Granular Urea (uniform granules)",
            "Neem-Coated Urea (slow release, 4-8 week)",
            "Sulfur-Coated Urea (enhanced durability)",
            "Ammonium Nitrate Urea (more soluble)"
        ],
        "brands": "Hindustan Urea Limited (HUL), KRIBHCO, Rashtriya Chemicals, IFFCO, Zuari",
        "price_indicator": "₹200-300 per 50kg bag (approximate)"
    },
    "DAP": {
        "name": "DAP - Diammonium Phosphate",
        "commercial_name": "Also known as: (NH4)2HPO4, Phosphate",
        "nutrient_composition": "Nitrogen (N): 18%, Phosphorus (P): 46%, NH4: 21%, PO4: 54%",
        "description": "DAP (Di-Ammonium Phosphate) contains 18% nitrogen and 46% phosphorus. It's excellent for root development, early plant growth, and seed formation.",
        "what_it_is": "DAP is a compound fertilizer made from ammonia and phosphoric acid. It's granular and highly soluble in water.",
        "benefits": [
            "Promotes strong root system development",
            "Enhances early plant growth and vigor",
            "Improves seed germination rates",
            "Facilitates flower and reproductive development",
            "Good dual source of Nitrogen and Phosphorus",
            "Particularly effective for young plants"
        ],
        "deficiency_symptoms": "Stunted root growth, delayed flowering, reduced seed set, purple-reddish leaves at harvest",
        "ideal_crops": ["Rice", "Wheat", "Maize", "Pulses", "Oilseeds", "Vegetables"],
        "application_timing": "At sowing time (basal application) for best results",
        "usage": "Apply 150-200 kg/hectare. Use as basal fertilizer at sowing time. Mix thoroughly with soil.",
        "dosage": {
            "Light soils": "150-175 kg/ha",
            "Medium soils": "175-200 kg/ha",
            "Heavy soils": "200-225 kg/ha"
        },
        "method": "Broadcast and mix with soil at sowing, or apply in furrows",
        "storage": "Keep in moisture-proof containers. Absorbs moisture in humid climates",
        "types": [
            "Granular DAP (uniform 1-3mm granules)",
            "Technical Grade DAP (industrial)",
            "Fortified DAP (with micronutrients - Zn, B, Mn)",
            "Coated DAP (slow-release coating)",
            "Soluble DAP solution"
        ],
        "brands": "Hindustan Urea Limited, KRIBHCO, Boron, Rashtriya, Chambal",
        "price_indicator": "₹300-400 per 50kg bag (approximate)"
    },
    "NPK": {
        "name": "NPK - Compound Fertilizer",
        "commercial_name": "Also known as: Balanced Fertilizer, Complex Fertilizer",
        "nutrient_composition": "Variable - Common ratios: 10:26:26, 12:32:16, 19:19:19, 13:40:13",
        "description": "NPK fertilizer contains balanced proportions of Nitrogen, Phosphorus, and Potassium in various ratios. Ideal for overall crop growth, balanced nutrition, and maximum yield.",
        "what_it_is": "NPK is a compound fertilizer containing all three primary macronutrients in one product. Perfect for farmers wanting single-bag application.",
        "benefits": [
            "Provides balanced nutrition for holistic growth",
            "Promotes overall plant development",
            "Improves crop health and immunity",
            "Enhances flower and fruit development",
            "Increases yield quantity and quality",
            "Suitable for diverse crop types",
            "Convenient single-bag application"
        ],
        "deficiency_symptoms": "Poor overall growth, reduced yield, weak plant structure, poor quality fruits",
        "ideal_crops": ["All crops", "Vegetables", "Fruits", "Cereals", "Pulses", "Sugarcane", "Cotton", "Oilseeds"],
        "application_timing": "50% at sowing, 50% during active growth stages",
        "usage": "Apply 250-350 kg/hectare in splits. Apply 50% at sowing, 50% during growth stages in 2-3 splits.",
        "dosage": {
            "Initial application": "50-150 kg/ha at sowing",
            "Top-dressing": "100-150 kg/ha at growth stage",
            "Total season": "250-350 kg/ha"
        },
        "method": "Mix with soil at sowing, apply as top-dressing during growth",
        "storage": "Keep in dry, cool well-ventilated store. Protect from moisture and rain",
        "types": [
            "10:26:26 NPK (Phosphorus, Potassium rich)",
            "12:32:16 NPK (Phosphorus-rich variant)",
            "19:19:19 NPK (Balanced 1:1:1 ratio)",
            "Customized NPK formulations (region-specific)",
            "Micro-nutrient fortified NPK"
        ],
        "common_ratios": {
            "10:26:26": "For phosphorus/potassium deficient soils",
            "12:32:16": "For cereals and pulse crops",
            "19:19:19": "Balanced for all crops",
            "13:40:13": "High phosphorus for flowering crops"
        },
        "brands": "KRIBHCO NPK, Rashtriya NPK, Novozymes, Chambal, Zuari, IFFCO",
        "price_indicator": "₹350-500 per 50kg bag (approximate)"
    },
    "SSP": {
        "name": "SSP - Single Super Phosphate",
        "commercial_name": "Also known as: Superphosphate, P-fertilizer",
        "nutrient_composition": "Phosphorus (P): 16%, Sulfur (S): 11-12%, Calcium: 20%",
        "description": "SSP (Single Super Phosphate) contains 16% phosphorus. It's a concentrated phosphorus source suitable for phosphorus-deficient soils and fruit/vegetable production.",
        "what_it_is": "SSP is produced by treating phosphate rock with sulfuric acid. It provides both phosphorus and sulfur, making it ideal for deficient soils.",
        "benefits": [
            "Strengthens root system development",
            "Promotes flower and fruit development",
            "Improves soil structure and tilth",
            "High phosphorus source for deficiency correction",
            "Contains sulfur for oil crop improvement",
            "Increases protein content in grains"
        ],
        "deficiency_symptoms": "Purple/reddish discoloration, stunted root growth, delayed flowering, poor seed formation",
        "ideal_crops": ["Pulses", "Oilseeds", "Vegetables", "Fruits", "Wheat", "Rice", "Cotton"],
        "application_timing": "As basal dose before sowing for maximum effect",
        "usage": "Apply 100-150 kg/hectare. Best applied as basal dose before sowing. Mix well with soil.",
        "dosage": {
            "P-deficient soils": "150-200 kg/ha",
            "Medium P soils": "100-150 kg/ha",
            "Pulse crops": "100-125 kg/ha"
        },
        "method": "Mix thoroughly with soil 2-3 weeks before sowing",
        "storage": "Keep in dry place, away from moisture. Can cake in humid conditions",
        "types": [
            "Granular SSP (50kg bag standard)",
            "Powder SSP (fine form)",
            "Enriched SSP (with micronutrients Zn, B)",
            "Water Soluble SSP (higher availability)"
        ],
        "brands": "IFFC, Indian Farmers Fertilizers, Chambal, Rashtriya, Zuari",
        "price_indicator": "₹250-350 per 50kg bag (approximate)"
    },
    "MOP": {
        "name": "MOP - Muriate of Potash",
        "commercial_name": "Also known as: Potassium Chloride, KCl, Potash",
        "nutrient_composition": "Potassium (K): 60% (as K2O), Chloride: 47%, Potassium: 49%",
        "description": "MOP (Muriate of Potash) is a potassium-rich fertilizer containing 60% potassium oxide (K2O). It's the most widely used potassium source globally for crop nutrition.",
        "what_it_is": "MOP is a naturally occurring mineral salt mined from deep geological deposits. It's highly concentrated in potassium, essential for plant growth.",
        "benefits": [
            "Enhances overall crop quality and appearance",
            "Improves disease and pest resistance",
            "Promotes stronger stems and roots",
            "Essential for fruit and vegetable crops",
            "Increases sugar content in fruits",
            "Improves shelf life of produce",
            "Strengthens plant immunity"
        ],
        "deficiency_symptoms": "Marginal leaf scorching (brown edges), weak stems, poor root development, reduced disease resistance",
        "ideal_crops": ["Potatoes", "Fruits", "Vegetables", "Sugarcane", "Tobacco", "Cotton", "Pulses"],
        "application_timing": "During active growth stage or pre-flowering for best results",
        "usage": "Apply 40-60 kg/hectare depending on crop and soil K status. Split application recommended for better absorption.",
        "dosage": {
            "High demand crops (potato, fruit)": "60-80 kg/ha",
            "Medium demand (cereals)": "40-60 kg/ha",
            "Pulses/oilseeds": "20-40 kg/ha"
        },
        "method": "Mix well with soil or apply through fertigation/drip irrigation",
        "storage": "Keep completely dry. Very hygroscopic - absorbs moisture readily. Use moisture-proof bags.",
        "types": [
            "Granular MOP (white/red crystals, 1-3mm)",
            "Crop-specific MOP blend (formulated for specific crops)",
            "Water Soluble MOP (100% soluble for fertigation)",
            "Coated MOP (slow-release, reduces K leaching)"
        ],
        "mahadhan_info": "Mahadhan MOP is manufactured under 'Bhartiya Jan Urvarak Pariyojana' subsidy scheme",
        "brands": "Karpco, KRIBHCO, Rashtriya, Indian Potash Limited, Mahadhan",
        "price_indicator": "₹350-450 per 50kg bag (approximate)"
    },
    "Zinc Sulphate": {
        "name": "Zinc Sulphate",
        "commercial_name": "Also known as: ZnSO4, Zinc Vitriol, White Vitriol, Micronutrient",
        "nutrient_composition": "Zinc (Zn): 21%, Sulfur (S): 11%, SO4: 49%",
        "description": "Zinc Sulphate is a micronutrient fertilizer containing 21% zinc and 11% sulfur. Essential for correcting zinc deficiency in soils, especially common in cereals and pulses.",
        "what_it_is": "Zinc Sulphate is a white crystalline compound. It's a readily available source of zinc for plants and is highly soluble in water.",
        "benefits": [
            "Improves overall crop yield significantly",
            "Enhances protein synthesis in plants",
            "Promotes enzyme activation and function",
            "Critical for cereals, pulses, and oilseeds",
            "Prevents stunted growth syndrome",
            "Improves grain quality and nutritional value",
            "Corrects zinc deficiency rapidly"
        ],
        "deficiency_symptoms": "Stunted growth, small leaves, chlorosis (yellowing), distorted leaf shape, poor grain filling",
        "deficiency_crops": ["Rice", "Wheat", "Maize (especially on alkaline soils)", "Pulses", "Oilseeds"],
        "ideal_crops": ["Cereals", "Pulses", "Oilseeds", "Rice", "Wheat", "Maize", "Beans", "Peas"],
        "application_timing": "At sowing time or early growth stage for best effect",
        "usage": "Apply 5-10 kg/hectare as soil application OR 0.5-1% as foliar spray. Best applied at sowing or early growth stages.",
        "dosage": {
            "Soil application": "5-10 kg/ha as single dose",
            "Foliar spray": "0.5-1% solution @ 500-800 L water/ha",
            "Deficient fields": "20-25 kg/ha in first year"
        },
        "method": {
            "Soil": "Mix with well-decomposed FYM, apply and irrigate",
            "Foliar": "Spray 0.5-1% solution at 3-4 leaf stage"
        },
        "storage": "Keep in tightly sealed containers to prevent moisture absorption and oxidation",
        "types": [
            "Monohydrate Zinc Sulphate (21% Zn) - single crystal form",
            "Heptahydrate Zinc Sulphate (commercial grade, 22% Zn)",
            "Chelated Zinc Sulphate (higher bioavailability)",
            "Zinc Sulphate with Boron blend (combined micronutrients)"
        ],
        "brands": "Century Enka, Ramakrishna Chemicals, Tronox, Agro Tech, Indian Minerals",
        "water_solubility": "Highly soluble in water - preferred for fertigation",
        "price_indicator": "₹500-700 per 50kg bag (approximate)"
    },
    "Compost": {
        "name": "Compost - Organic Fertilizer",
        "commercial_name": "Also known as: Farmyard Manure (FYM), Vermicompost, Organic Matter",
        "nutrient_composition": "Nitrogen (N): 0.5-2%, Phosphorus: 0.3-1.5%, Potassium: 0.5-1.5%, Rich Organic Matter",
        "description": "Compost is an organic fertilizer made from decomposed organic matter, farm waste, manure, and leaves. Contains 0.5-2% nitrogen and significantly improves soil structure.",
        "what_it_is": "Compost is decomposed organic matter rich in humus. It's sustainable, environmentally friendly, and improves long-term soil health.",
        "benefits": [
            "Improves soil fertility and nitrogen content",
            "Dramatically increases organic matter content",
            "Enhances water retention capacity",
            "Promotes beneficial soil microbes and microbial activity",
            "Improves soil structure and tilth",
            "Sustainable and environment-friendly farming option",
            "Reduces chemical fertilizer dependency",
            "Improves soil health for long-term productivity"
        ],
        "soil_improvement": [
            "Increases organic carbon from 1-2% to 2-3% annually",
            "Improves water holding capacity",
            "Better nutrient availability",
            "Reduces soil compaction",
            "Increases soil porosity"
        ],
        "ideal_crops": ["All crops benefit from compost", "Vegetables (especially leafy)", "Fruits", "Flowers", "Nursery plants"],
        "application_timing": "2-3 weeks before planting for decomposition and nutrient release",
        "usage": "Apply 5-10 tons/hectare annually. Mix well with soil 2-3 weeks before planting. Can be used every season for continuous soil conditioning.",
        "dosage": {
            "Soil building": "10 tons/ha annually for 3-4 years",
            "Maintenance": "5-7 tons/ha annually after initial buildup",
            "Container gardening": "30-40% of potting mix"
        },
        "method": "Mix thoroughly with top 15-20cm soil layer, irrigate well for settling",
        "storage": "Store in well-drained area. Cover to prevent rain leaching. Cure for 2-3 months for maturity.",
        "types": [
            "Farm Yard Manure (FYM) - dried animal manure + farm waste",
            "Vermicompost (worm compost) - highest quality, enzyme rich, fastest acting",
            "Kitchen waste compost - made from vegetable scraps",
            "Coir pith compost - coconut fiber based, excellent water retention",
            "Neem cake compost - with neem leaf supplementation"
        ],
        "vermicompost_advantages": [
            "Contains beneficial microorganisms",
            "Rich in plant hormones and enzymes",
            "Faster nutrient availability",
            "Better disease suppression",
            "Higher cost-benefit for vegetables/horticulture"
        ],
        "brands": "Bio-gold, Gomukh, Organic India, Vriksha, Evergreen Compost",
        "sustainability": "Reduces chemical fertilizer use by 50-60% over 3-4 years",
        "return_on_investment": "3-5 year payback period through improved yields and reduced input costs",
        "price_indicator": "₹150-250 per 50kg bag for FYM, ₹300-500 for Vermicompost (approximate)"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_image_path(fertilizer_name):
    """Get local image path for fertilizer. Add your images here manually."""
    image_dir = f"images/{fertilizer_name}"
    
    # Check if directory exists
    if os.path.exists(image_dir):
        # Look for image files (jpg, png, jpeg)
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                return os.path.join(image_dir, filename)
    
    return None  # Return None if no image found


def display_fertilizer_image(fertilizer_name):
    """Display fertilizer image with fallback to description."""
    image_path = get_image_path(fertilizer_name)
    
    if image_path and os.path.exists(image_path):
        # Image found - display it
        st.image(image_path, use_container_width=True, caption=f"{fertilizer_name} Fertilizer")
        return True
    else:
        # No image - show placeholder
        st.markdown(
            f"""
            <div style='background-color: #1a1a1a; border: 2px dashed #333333; 
                        border-radius: 12px; padding: 2rem; text-align: center;'>
            <p style='color: #aaaaaa; font-size: 1rem; margin: 1rem 0;'>
            📸 <strong>{fertilizer_name} Product Image</strong></p>
            <p style='color: #888888; font-size: 0.9rem;'>
            Add image to: <code>images/{fertilizer_name}/</code>
            </p>
            <p style='color: #666666; font-size: 0.85rem;'>
            Supported formats: JPG, PNG, JPEG, WEBP
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return False


def render_custom_css():
    """Apply custom CSS styling to the app."""
    st.markdown(custom_css, unsafe_allow_html=True)


def load_ml_models():
    """Load trained ML models and encoders from disk."""
    try:
        model = joblib.load('fertilizer_model.pkl')
        crop_encoder = joblib.load('crop_encoder.pkl')
        fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, crop_encoder, fertilizer_encoder, scaler
    except FileNotFoundError:
        st.warning("⚠️ Models not found. Please run train_model.py first.")
        return None, None, None, None


def predict_fertilizer(N, P, K, pH, crop, model, crop_encoder, fertilizer_encoder, scaler):
    """
    Make fertilizer prediction using ML model.
    
    Parameters:
    -----------
    N, P, K, pH : float
        Soil nutrient and pH values
    crop : str
        Selected crop name
    model, crop_encoder, fertilizer_encoder, scaler : objects
        ML pipeline objects
    
    Returns:
    --------
    fertilizer : str
        Recommended fertilizer type
    confidence : float
        Prediction confidence (70-95%)
    """
    try:
        if model is None:
            return "NPK Balanced", 75
        
        # Encode crop
        crop_encoded = crop_encoder.transform([crop])[0]
        
        # Create feature array
        features = np.array([[N, P, K, pH, crop_encoded]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability for confidence
        prediction_proba = model.predict_proba(features_scaled)[0]
        confidence = int(np.max(prediction_proba) * 100)
        
        # Decode fertilizer
        fertilizer = fertilizer_encoder.inverse_transform([prediction])[0]
        
        return fertilizer, confidence
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "NPK Balanced", 75


def display_npk_analysis(N, P, K):
    """Display NPK values with deficiency/excess highlighting."""
    col1, col2, col3 = st.columns(3)
    
    # Define normal ranges
    n_status = "High" if N > 200 else "Low" if N < 100 else "Normal"
    p_status = "High" if P > 100 else "Low" if P < 30 else "Normal"
    k_status = "High" if K > 300 else "Low" if K < 100 else "Normal"
    
    with col1:
        st.metric(
            "Nitrogen (N)",
            f"{N} ppm",
            delta=n_status,
            delta_color="inverse" if n_status == "Normal" else "off"
        )
    
    with col2:
        st.metric(
            "Phosphorus (P)",
            f"{P} ppm",
            delta=p_status,
            delta_color="inverse" if p_status == "Normal" else "off"
        )
    
    with col3:
        st.metric(
            "Potassium (K)",
            f"{K} ppm",
            delta=k_status,
            delta_color="inverse" if k_status == "Normal" else "off"
        )


def create_npk_chart(N, P, K):
    """Create a bar chart for NPK values."""
    fig = go.Figure(data=[
        go.Bar(x=['N', 'P', 'K'], y=[N, P, K],
               marker_color=['#FF9500', '#10b981', '#00a8d8'],
               text=[f'{N} ppm', f'{P} ppm', f'{K} ppm'],
               textposition='auto',
               showlegend=False)
    ])
    
    fig.update_layout(
        title="Soil NPK Composition",
        xaxis_title="Nutrient",
        yaxis_title="PPM (Parts Per Million)",
        template="plotly_dark",
        height=300,
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig


def display_fertilizer_details(fertilizer_type):
    """Display essential fertilizer information in 5 key points with vibrant colors and readable text."""
    if fertilizer_type not in FERTILIZER_DATABASE:
        st.warning(f"Information not available for {fertilizer_type}")
        return
    
    info = FERTILIZER_DATABASE[fertilizer_type]
    
    # Create 2-column layout: Image (Left) | Details (Right)
    col_image, col_details = st.columns([1, 1.5])
    
    # ======================= COLUMN 1: IMAGE =======================
    with col_image:
        st.subheader("📸 Product")
        display_fertilizer_image(fertilizer_type)
        
        # Basic Info Card
        st.markdown("---")
        st.markdown(f"### 🌾 {info.get('name', fertilizer_type)}")
        if "commercial_name" in info:
            st.caption(f"_{info['commercial_name']}_")
    
    # ======================= COLUMN 2: 5 KEY POINTS WITH VIBRANT COLORS =======================
    with col_details:
        st.subheader("📋 Key Information")
        
        # POINT 1: Nutrient Composition (BRIGHT CYAN)
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #00bcd4 0%, #0097a7 100%); 
                        padding: 15px; border-left: 6px solid #00838f; border-radius: 8px; 
                        margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                <p style='color: #ffffff; margin: 0; font-size: 18px; font-weight: bold;'>🧪 1. What Nutrients Does It Contain?</p>
                <p style='color: #ffffff; margin: 10px 0 0 0; font-weight: bold; font-size: 16px;'>{info.get('nutrient_composition', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        # POINT 2: What Is It (BRIGHT LIME GREEN)
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #76ff03 0%, #558b2f 100%); 
                        padding: 15px; border-left: 6px solid #33691e; border-radius: 8px; 
                        margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                <p style='color: #1b5e20; margin: 0; font-size: 18px; font-weight: bold;'>❓ 2. What Is It & How Does It Help in Crop Nutrition?</p>
                <p style='color: #1b5e20; margin: 10px 0 0 0; font-size: 15px; font-weight: 500;'>{info.get('what_it_is', info.get('description', 'N/A'))}</p>
            </div>
            """, unsafe_allow_html=True
        )
        
        # POINT 3: Benefits for Farmers (BRIGHT ORANGE)
        benefits_list = ""
        if isinstance(info.get("benefits"), list):
            for benefit in info["benefits"][:5]:
                benefits_list += f"<li style='color: #bf360c; margin: 8px 0; font-size: 15px; font-weight: 500;'>{benefit}</li>"
        else:
            benefits_list = f"<p style='color: #bf360c; font-weight: 500;'>{info.get('benefits', 'N/A')}</p>"
        
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #ffb74d 0%, #ff9800 100%); 
                        padding: 15px; border-left: 6px solid #e65100; border-radius: 8px; 
                        margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                <p style='color: #bf360c; margin: 0; font-size: 18px; font-weight: bold;'>💚 3. How Does It Benefit Farmers?</p>
                <ul style='margin: 10px 0 0 20px; padding: 0;'>{benefits_list}</ul>
            </div>
            """, unsafe_allow_html=True
        )
        
        # POINT 4: Ideal Crops (BRIGHT PURPLE)
        if "ideal_crops" in info:
            crops = info["ideal_crops"]
            crops_text = ", ".join(crops)
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #ba68c8 0%, #8e24aa 100%); 
                            padding: 15px; border-left: 6px solid #6a1b9a; border-radius: 8px; 
                            margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                    <p style='color: #ffffff; margin: 0; font-size: 18px; font-weight: bold;'>🌽 4. What Are the Crops in Which Farmers Could Use It?</p>
                    <p style='color: #ffffff; margin: 10px 0 0 0; font-weight: 600; font-size: 15px;'>{crops_text}</p>
                </div>
                """, unsafe_allow_html=True
            )
        
        # POINT 5: Application & Dosage (BRIGHT PINK/RED)
        timing_text = info.get('application_timing', 'N/A')
        usage_text = info.get('usage', 'N/A')
        try:
            first_line = str(usage_text).split('.')[0] if usage_text else 'N/A'
        except:
            first_line = str(usage_text)
        
        col_timing, col_dosage = st.columns(2)
        
        with col_timing:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #ff69b4 0%, #e91e63 100%); 
                            padding: 12px; border-left: 6px solid #c2185b; border-radius: 8px; 
                            margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                    <p style='color: #ffffff; margin: 0; font-size: 14px; font-weight: bold;'>🌾 When to Apply</p>
                    <p style='color: #ffffff; margin: 8px 0 0 0; font-weight: bold; font-size: 13px;'>{timing_text}</p>
                </div>
                """, unsafe_allow_html=True
            )
        
        with col_dosage:
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #ff6b6b 0%, #f44336 100%); 
                            padding: 12px; border-left: 6px solid #d32f2f; border-radius: 8px; 
                            margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);'>
                    <p style='color: #ffffff; margin: 0; font-size: 14px; font-weight: bold;'>📊 Recommended Dosage</p>
                    <p style='color: #ffffff; margin: 8px 0 0 0; font-weight: bold; font-size: 13px;'>{first_line}</p>
                </div>
                """, unsafe_allow_html=True
            )


def render_sidebar():
    """Render sidebar with information."""
    with st.sidebar:
        st.markdown("### 📊 Dashboard Info")
        st.write("This AI-powered system recommends fertilizers based on soil analysis.")
        
        st.divider()
        
        st.markdown("### 🌾 About This System")
        st.write("""
        - Uses machine learning to analyze soil composition
        - Provides personalized fertilizer recommendations
        - Considers NPK levels, pH, and crop type
        - Displays detailed fertilizer information
        """)
        
        st.divider()
        
        st.markdown("### 📝 How It Works")
        st.write("""
        1. Enter your soil test values
        2. Select your primary crop
        3. Click "Analyze & Recommend"
        4. Get personalized fertilizer suggestion
        5. View detailed information about the recommended fertilizer
        """)
        
        st.divider()
        
        st.markdown("### 💡 Pro Tips")
        st.write("""
        - Get soil tested from certified lab
        - Consider crop rotation patterns
        - Follow recommended application rates
        - Monitor soil regularly
        """)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function."""
    
    # Apply custom CSS
    render_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Load ML models
    model, crop_encoder, fertilizer_encoder, scaler = load_ml_models()
    
    # Get available crops
    if crop_encoder is not None:
        available_crops = sorted(list(crop_encoder.classes_))
    else:
        available_crops = ["Rice", "Wheat", "Maize", "Cotton", "Pulses", "Sugarcane", 
                          "Barley", "Soybean", "Groundnut", "Chickpea", "Tobacco", "Oats", 
                          "Millet", "Lentil", "Jute"]
    
    # Main Title
    st.markdown(
        "<h1 class='main-title'>🌾 Precision Nutrient Management</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p class='subtitle'>Get customized fertilizer recommendations based on your soil's NPK levels and pH</p>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Initialize session state
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "recommendation_data" not in st.session_state:
        st.session_state.recommendation_data = None
    
    # Two-column layout
    col_input, col_result = st.columns([1, 1.4], gap="large")
    
    # =====================================================================
    # LEFT COLUMN - INPUT FORM
    # =====================================================================
    
    with col_input:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        
        st.markdown("### 🧪 Soil Analysis Input")
        
        # Nitrogen input
        st.markdown("<label class='input-label'>Nitrogen (N) - ppm</label>", unsafe_allow_html=True)
        nitrogen = st.number_input(
            "Nitrogen",
            min_value=0,
            max_value=500,
            value=280,
            step=10,
            label_visibility="collapsed",
            key="nitrogen_input"
        )
        st.markdown("<p class='helper-text'>Available nitrogen in soil</p>", unsafe_allow_html=True)
        
        # Phosphorus input
        st.markdown("<label class='input-label'>Phosphorus (P) - ppm</label>", unsafe_allow_html=True)
        phosphorus = st.number_input(
            "Phosphorus",
            min_value=0,
            max_value=200,
            value=45,
            step=5,
            label_visibility="collapsed",
            key="phosphorus_input"
        )
        st.markdown("<p class='helper-text'>Available phosphorus in soil</p>", unsafe_allow_html=True)
        
        # Potassium input
        st.markdown("<label class='input-label'>Potassium (K) - ppm</label>", unsafe_allow_html=True)
        potassium = st.number_input(
            "Potassium",
            min_value=0,
            max_value=500,
            value=220,
            step=10,
            label_visibility="collapsed",
            key="potassium_input"
        )
        st.markdown("<p class='helper-text'>Available potassium in soil</p>", unsafe_allow_html=True)
        
        # pH input
        st.markdown("<label class='input-label'>Soil pH</label>", unsafe_allow_html=True)
        soil_ph = st.number_input(
            "pH",
            min_value=4.0,
            max_value=9.0,
            value=6.5,
            step=0.1,
            label_visibility="collapsed",
            key="ph_input"
        )
        st.markdown("<p class='helper-text'>Soil pH level (4.0-9.0)</p>", unsafe_allow_html=True)
        
        # Organic matter input
        st.markdown("<label class='input-label'>Organic Matter (%)</label>", unsafe_allow_html=True)
        organic_matter = st.number_input(
            "Organic Matter",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1,
            label_visibility="collapsed",
            key="om_input"
        )
        st.markdown("<p class='helper-text'>Percentage of organic matter</p>", unsafe_allow_html=True)
        
        # Crop selection
        st.markdown("<label class='input-label'>Select Crop</label>", unsafe_allow_html=True)
        selected_crop = st.selectbox(
            "Choose your primary crop",
            options=available_crops,
            index=0,
            label_visibility="collapsed",
            key="crop_input"
        )
        st.markdown("<p class='helper-text'>Primary crop to be grown</p>", unsafe_allow_html=True)
        
        # Analyze button
        if st.button("🔍 Analyze & Recommend", use_container_width=True):
            # Make prediction
            fertilizer, confidence = predict_fertilizer(
                nitrogen, phosphorus, potassium, soil_ph, selected_crop,
                model, crop_encoder, fertilizer_encoder, scaler
            )
            
            # Store results in session state
            st.session_state.show_results = True
            st.session_state.recommendation_data = {
                "fertilizer": fertilizer,
                "confidence": confidence,
                "N": nitrogen,
                "P": phosphorus,
                "K": potassium,
                "pH": soil_ph,
                "OM": organic_matter,
                "crop": selected_crop
            }
            st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # =====================================================================
    # RIGHT COLUMN - RESULTS DASHBOARD
    # =====================================================================
    
    with col_result:
        with st.container(border=True):
            if st.session_state.show_results and st.session_state.recommendation_data:
                data = st.session_state.recommendation_data
                
                # Success message
                st.success("✓ Analysis completed successfully!")
                
                # Recommended fertilizer
                st.markdown(
                    f"<div class='success-box'>"
                    f"<strong>🌾 Recommended Fertilizer:</strong><br/>"
                    f"<div class='fertilizer-name'>{data['fertilizer']}</div>"
                    f"<p style='text-align: center; color: #10b981;'>"
                    f"Confidence: <strong>{data['confidence']}%</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                st.divider()
                
                # Display fertilizer image (local or placeholder)
                display_fertilizer_image(data['fertilizer'])
                
                st.divider()
                
                # NPK Analysis
                st.subheader("📊 Soil Composition Analysis")
                display_npk_analysis(data['N'], data['P'], data['K'])
                
                st.divider()
                
                # NPK Chart
                st.plotly_chart(create_npk_chart(data['N'], data['P'], data['K']), use_container_width=True)
                
                st.divider()
                
                # Soil Summary
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Soil pH", f"{data['pH']}", "Neutral" if 6.0 <= data['pH'] <= 7.5 else "Acidic/Alkaline")
                with col_s2:
                    st.metric("Organic Matter", f"{data['OM']}%")
                
                st.divider()
                
                # New Analysis button
                if st.button("↻ New Analysis", use_container_width=True):
                    st.session_state.show_results = False
                    st.session_state.recommendation_data = None
                    st.rerun()
            
            else:
                # Default state - Ready for analysis
                st.markdown(
                    """
                    <div style='text-align: center; padding: 4rem 2rem;'>
                    <div style='font-size: 4rem; margin-bottom: 1rem;'>🔬</div>
                    <h3 style='color: #00a8d8; font-size: 1.8rem;'>Ready for Analysis</h3>
                    <p style='color: #cccccc; font-size: 1rem; line-height: 1.6; margin-top: 1rem;'>
                    Enter your soil test values on the left and click "Analyze & Recommend" to get personalized fertilizer recommendations.
                    </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # =====================================================================
    # FERTILIZER DETAILS SECTION (Full Width)
    # =====================================================================
    
    st.divider()
    
    if st.session_state.show_results and st.session_state.recommendation_data:
        data = st.session_state.recommendation_data
        
        st.markdown("## 📚 Detailed Fertilizer Information")
        
        with st.container():
            display_fertilizer_details(data['fertilizer'])
        
        st.divider()
        
        # Additional recommendations
        st.markdown("## 💡 Additional Recommendations")
        
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown(
                """
                <div class='info-section'>
                <strong>✓ Best Time to Apply</strong><br/>
                Apply during early growth stages or as per crop calendar.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_rec2:
            st.markdown(
                """
                <div class='info-section'>
                <strong>✓ Application Method</strong><br/>
                Mix with soil or dilute in irrigation water as per instructions.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_rec3:
            st.markdown(
                """
                <div class='info-section'>
                <strong>✓ Monitoring</strong><br/>
                Monitor crop health and retest soil after 60-90 days.
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # =====================================================================
    # FOOTER - Simple About Section
    # =====================================================================
    
    st.divider()
    
    footer_content = (
        "<div style='text-align: center; padding: 2rem; "
        "background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); "
        "border-radius: 12px; margin-top: 2rem; border: 2px solid #00ff88;'>"
        
        "<h2 style='color: #00ff88; font-size: 1.8rem; margin-bottom: 0.5rem; "
        "text-transform: uppercase; letter-spacing: 2px; font-weight: bold;'>"
        "🌾 About This Application</h2>"
        
        "<h3 style='color: #00d4ff; font-size: 1.2rem; margin: 0.8rem 0;'>"
        "AI-Powered Fertilizer Recommendation System</h3>"
        
        "<p style='color: #aabbcc; font-size: 0.95rem; line-height: 1.5; margin: 1rem auto; "
        "max-width: 900px;'>"
        "An intelligent ML application that analyzes soil composition and recommends "
        "optimal fertilizer combinations for maximum crop yield and soil health.</p>"
        
        "<div style='background: linear-gradient(135deg, rgba(0, 255, 136, 0.15), rgba(0, 212, 255, 0.15)); "
        "border: 2px solid #00ff88; padding: 1.2rem; margin: 1.2rem auto; max-width: 800px; border-radius: 8px;'>"
        "<p style='color: #00ff88; font-weight: bold; margin: 0; font-size: 0.95rem;'>"
        "✨ 88.89% Accuracy | Random Forest, Decision Tree, SVM, XGBoost | 8313+ Samples</p>"
        "</div>"
        
        "<p style='color: #aabbcc; font-size: 0.9rem; margin: 1rem 0;'>"
        "🛠️ Built with: Streamlit • scikit-learn • XGBoost • Pandas • NumPy • Plotly</p>"
        
        "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; "
        "margin: 1.2rem 0; max-width: 800px; margin-left: auto; margin-right: auto;'>"
        "<div style='background: #ff6b6b; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>🌾 Urea</div>"
        "<div style='background: #4ecdc4; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>🌱 DAP</div>"
        "<div style='background: #ffd93d; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>🎯 NPK</div>"
        "<div style='background: #aa96da; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>✨ SSP</div>"
        "<div style='background: #f38181; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>💎 MOP</div>"
        "<div style='background: #95e1d3; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>⚡ Zinc</div>"
        "<div style='background: #f78fb3; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>🌍 Compost</div>"
        "<div style='background: #667eea; padding: 0.5rem; border-radius: 6px; color: white; "
        "font-weight: bold; font-size: 0.8rem;'>🔬 AI Ready</div>"
        "</div>"
        
        "<p style='color: #666677; font-size: 0.8rem; margin-top: 1rem;'>"
        "© 2026 Fertilizer Recommendation AI | All Rights Reserved</p>"
        "</div>"
    )
    
    st.markdown(footer_content, unsafe_allow_html=True)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
