# Crop Disease Database - All 38 PlantVillage Classes
# Keys match tensorflow_datasets 'plant_village' label names exactly

DISEASE_DB = {
    "Apple___Apple_scab": {
        "plant": "Apple", "botanical_name": "Malus domestica",
        "disease": "Apple Scab",
        "treatment": "Apply sulfur or copper-based fungicides. Remove and destroy fallen leaves.",
        "prevention": "Plant resistant varieties. Prune to improve airflow."
    },
    "Apple___Black_rot": {
        "plant": "Apple", "botanical_name": "Malus domestica",
        "disease": "Apple Black Rot",
        "treatment": "Prune out infected branches. Apply Captan or thiophanate-methyl fungicide.",
        "prevention": "Remove mummified fruit and maintain tree hygiene."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple", "botanical_name": "Malus domestica",
        "disease": "Cedar Apple Rust",
        "treatment": "Apply fungicides (myclobutanil) during bud break and bloom.",
        "prevention": "Remove nearby juniper/cedar trees. Plant resistant cultivars."
    },
    "Apple___healthy": {
        "plant": "Apple", "botanical_name": "Malus domestica",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed. Continue regular monitoring.",
        "prevention": "Maintain proper irrigation and balanced fertilization."
    },
    "Blueberry___healthy": {
        "plant": "Blueberry", "botanical_name": "Vaccinium corymbosum",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Keep soil pH 4.5–5.5. Mulch to retain moisture."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry", "botanical_name": "Prunus avium",
        "disease": "Powdery Mildew",
        "treatment": "Apply sulfur or potassium bicarbonate. Increase air circulation.",
        "prevention": "Avoid excessive nitrogen. Prune for airflow."
    },
    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry", "botanical_name": "Prunus avium",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Regular pruning and disease monitoring."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn", "botanical_name": "Zea mays",
        "disease": "Gray Leaf Spot",
        "treatment": "Apply foliar fungicides (azoxystrobin). Rotate crops.",
        "prevention": "Use resistant hybrids and bury infected residue."
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn", "botanical_name": "Zea mays",
        "disease": "Common Rust",
        "treatment": "Apply fungicides (propiconazole) if severe early infection.",
        "prevention": "Plant resistant hybrids. Scout regularly."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn", "botanical_name": "Zea mays",
        "disease": "Northern Leaf Blight",
        "treatment": "Fungicides (mancozeb) applied at disease onset.",
        "prevention": "Crop rotation, resistant varieties, plow in infected residue."
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn", "botanical_name": "Zea mays",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Scout for pests. Ensure adequate soil nutrients."
    },
    "Grape___Black_rot": {
        "plant": "Grape", "botanical_name": "Vitis vinifera",
        "disease": "Black Rot",
        "treatment": "Spray mancozeb or myclobutanil. Remove infected berries and mummies.",
        "prevention": "Improve canopy airflow. Remove leaf debris."
    },
    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape", "botanical_name": "Vitis vinifera",
        "disease": "Esca (Black Measles)",
        "treatment": "No cure; remove infected vines. Use trunk wound protectants.",
        "prevention": "Protect pruning wounds immediately with fungicide paste."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape", "botanical_name": "Vitis vinifera",
        "disease": "Isariopsis Leaf Spot",
        "treatment": "Apply copper-based fungicides. Remove infected leaves.",
        "prevention": "Avoid leaf wetness; improve drainage and spacing."
    },
    "Grape___healthy": {
        "plant": "Grape", "botanical_name": "Vitis vinifera",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Monitor for powdery mildew and downy mildew."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange", "botanical_name": "Citrus sinensis",
        "disease": "Huanglongbing (Citrus Greening)",
        "treatment": "No cure. Remove and destroy infected trees. Control ACP psyllid.",
        "prevention": "Use certified disease-free nursery stock. Monitor for psyllids."
    },
    "Peach___Bacterial_spot": {
        "plant": "Peach", "botanical_name": "Prunus persica",
        "disease": "Bacterial Spot",
        "treatment": "Apply copper sprays in spring. Prune to improve airflow.",
        "prevention": "Plant resistant varieties. Avoid wounding."
    },
    "Peach___healthy": {
        "plant": "Peach", "botanical_name": "Prunus persica",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Annual pruning and regular disease scouting."
    },
    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper", "botanical_name": "Capsicum annuum",
        "disease": "Bacterial Spot",
        "treatment": "Apply copper bactericide. Remove affected plant parts.",
        "prevention": "Use pathogen-free seed. Avoid overhead irrigation."
    },
    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper", "botanical_name": "Capsicum annuum",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Rotate crops annually to prevent soil buildup of pathogens."
    },
    "Potato___Early_blight": {
        "plant": "Potato", "botanical_name": "Solanum tuberosum",
        "disease": "Early Blight",
        "treatment": "Apply chlorothalonil or mancozeb. Remove infected lower leaves.",
        "prevention": "Crop rotation; avoid overhead irrigation."
    },
    "Potato___Late_blight": {
        "plant": "Potato", "botanical_name": "Solanum tuberosum",
        "disease": "Late Blight",
        "treatment": "Apply fungicides immediately (cymoxanil + mancozeb). Destroy infected plants.",
        "prevention": "Use certified seed. Avoid planting near tomatoes."
    },
    "Potato___healthy": {
        "plant": "Potato", "botanical_name": "Solanum tuberosum",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Hill plants properly. Ensure soil drainage."
    },
    "Raspberry___healthy": {
        "plant": "Raspberry", "botanical_name": "Rubus idaeus",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Prune old canes after harvest. Train plants for airflow."
    },
    "Soybean___healthy": {
        "plant": "Soybean", "botanical_name": "Glycine max",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Practice crop rotation with non-legumes."
    },
    "Squash___Powdery_mildew": {
        "plant": "Squash", "botanical_name": "Cucurbita pepo",
        "disease": "Powdery Mildew",
        "treatment": "Apply neem oil, potassium bicarbonate, or sulfur-based fungicides.",
        "prevention": "Space plants wide apart. Avoid evening watering."
    },
    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry", "botanical_name": "Fragaria × ananassa",
        "disease": "Leaf Scorch",
        "treatment": "Apply fungicides (captan). Remove and destroy old infected leaves.",
        "prevention": "Use resisttant varieties. Drip irrigate, not overhead."
    },
    "Strawberry___healthy": {
        "plant": "Strawberry", "botanical_name": "Fragaria × ananassa",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Remove old leaves in winter renovation."
    },
    "Tomato___Bacterial_spot": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Bacterial Spot",
        "treatment": "Spray copper-based bactericide. Remove infected plants.",
        "prevention": "Use pathogen-free seeds. Avoid working in wet foliage."
    },
    "Tomato___Early_blight": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Early Blight",
        "treatment": "Apply mancozeb or chlorothalonil. Remove lower infected leaves.",
        "prevention": "Stake plants, rotate crops, mulch around base."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Late Blight",
        "treatment": "Apply fungicides immediately. Destroy infected plants.",
        "prevention": "Avoid planting near potatoes. Choose resistant varieties."
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Leaf Mold",
        "treatment": "Improve ventilation. Apply copper fungicide.",
        "prevention": "Reduce humidity in greenhouses. Remove infected leaves."
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Septoria Leaf Spot",
        "treatment": "Apply mancozeb or copper fungicide. Remove infected lower leaves.",
        "prevention": "Mulch soil, rotate crops, avoid leaf wetness."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Spider Mites (Two-spotted)",
        "treatment": "Apply miticides or insecticidal soap. Water plants to reduce dust.",
        "prevention": "Avoid over-fertilizing with nitrogen. Introduce predatory mites."
    },
    "Tomato___Target_Spot": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Target Spot",
        "treatment": "Apply azoxystrobin or pyraclostrobin fungicides.",
        "prevention": "Crop rotation. Remove infected plant debris."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Yellow Leaf Curl Virus",
        "treatment": "No cure. Remove infected plants. Control whitefly vectors.",
        "prevention": "Use virus-resistant varieties. Use reflective mulch to deter whiteflies."
    },
    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Tomato Mosaic Virus",
        "treatment": "No cure. Remove infected plants. Disinfect tools.",
        "prevention": "Wash hands before handling. Use resistant varieties."
    },
    "Tomato___healthy": {
        "plant": "Tomato", "botanical_name": "Solanum lycopersicum",
        "disease": "Healthy Leaf",
        "treatment": "No treatment needed.",
        "prevention": "Practice crop rotation and consistent watering."
    },
}

# Ordered list of labels matching tensorflow_datasets 'plant_village' class order
CLASS_NAMES = list(DISEASE_DB.keys())

# Model Settings
MODEL_INPUT_SHAPE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4
SAVED_MODEL_PATH = "saved_model/plantvillage_mobilenetv2"

# UI Settings
APP_TITLE = "AgroGuard AI: Crop Disease Identifier"
THEME_COLOR = "#2E7D32"
