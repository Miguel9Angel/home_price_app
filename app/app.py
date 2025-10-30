import streamlit as st 
import pandas as pd
import pydeck as pdk

data = pd.read_csv('../data/apartments_bogota.csv')
st.title('Apartment Price Prediction')

price_min, price_max = data["rent_price"].min(), data["rent_price"].max()
data["color_value"] = ((data["rent_price"] - price_min) / (price_max - price_min) * 255).astype(int)

data["color"] = data["color_value"].apply(lambda x: [x, 255 - x, 100, 180])

view_state = pdk.ViewState(
    latitude=data['latitude'].mean(),
    longitude=data['longitude'].mean(),
    zoom=12,
    pitch=50,
)

layer = pdk.Layer(
    "ColumnLayer",
    data,
    get_position='[longitude, latitude]',
    get_elevation='constructed_area',   # ← altura proporcional al área
    elevation_scale=0.25,                # ← ajusta el factor de escala
    radius=50,
    get_fill_color='color',
    pickable=True,
    extruded=True
)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Localidad}\n${rent_price} COP\n${constructed_area} m2"}))

st.title("🏠 Apartment Data Entry Form")

st.write("Please fill in the details below to add a new apartment record.")

with st.form("apartment_form"):
    longitude = st.number_input("Longitude", format="%.6f", step=0.000001)
    latitude = st.number_input("Latitude", format="%.6f", step=0.000001)
    stratum = st.number_input("Stratum", min_value=1, max_value=6, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
    constructed_area = st.number_input("Constructed Area (m²)", min_value=10, step=1)
    house_age = st.number_input("House Age (years)", min_value=0, step=1)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
    parking = st.number_input("Number of Parking Spots", min_value=0, step=1)
    administration = st.number_input("Administration Fee (COP)", min_value=0, step=50000)
    floor = st.number_input("Floor", min_value=1, step=1)

    submitted = st.form_submit_button("Submit")

if submitted:
    st.success("✅ Apartment data successfully submitted!")

    new_data = {
        "longitude": longitude,
        "latitude": latitude,
        "stratum": stratum,
        "bathrooms": bathrooms,
        "constructed_area": constructed_area,
        "house_age": house_age,
        "bedrooms": bedrooms,
        "parking": parking,
        "administration": administration,
        "floor": floor
    }

    st.write("### 📋 Entered Data:")
    st.dataframe(pd.DataFrame([new_data]))
