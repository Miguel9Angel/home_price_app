import streamlit as st 
import pandas as pd
import pydeck as pdk
from geopy.geocoders import Nominatim
import requests
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

model_filename = '../model/best_xgb_pipeline.pkl'

try:
    loaded_model = joblib.load(model_filename)
    print(f"Model '{model_filename}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{model_filename}' was not found. Check the path.")
    exit()

data = pd.read_csv('../data/apartments_bogota.csv')
apto_data = data.copy()

tab1, tab2, tab3 = st.tabs(["🏠 Make a prediction", "📊 Market Analisys (EDA)", "🧠 Methodology"])

with tab1:
    st.title('Apartment Price Prediction')

    price_min, price_max = data["rent_price"].min(), data["rent_price"].max()
    data["color_value"] = ((data["rent_price"] - price_min) / (price_max - price_min) * 255).astype(int)

    data["color"] = data["color_value"].apply(lambda x: [x, 255 - x, 100, 180])

    options_dict = {
        "Stratum (Socioeconomic level)": "stratum",
        "Constructed Area (m²)": "constructed_area",
        "House Age (years)": "house_age",
        "Number of Parking Spots": "parking",
        "Rent Price (COP)": "rent_price"
    }

    option_label = st.selectbox(
        'Height representation of the points.',
        list(options_dict.keys())
    )
    option_height_map = options_dict[option_label]

    min_val = data[option_height_map].min()
    max_val = data[option_height_map].max()
    data['normalized_height'] = 100*(data[option_height_map]-min_val)/(max_val-min_val+1e-8)

    view_state = pdk.ViewState(
        latitude=data['latitude'].mean(),
        longitude=data['longitude'].mean(),
        zoom=10,
        pitch=50,
    )

    layer = pdk.Layer(
        "ColumnLayer",
        data,
        get_position='[longitude, latitude]',
        get_elevation='normalized_height',   
        elevation_scale=5,               
        radius=25,
        get_fill_color='color',
        pickable=True,
        extruded=True
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Localidad}\n${rent_price} COP\n{constructed_area} m2\nstratum {stratum}"}))

    st.write("## 🏠 Predict the fair rent price of your aparment")

    st.write("Please fill in the details below to add a new apartment record.")

    st.write('You can easily find the latitude and longitude by entering your addres into Google' \
                'Maps and rigth-clicking on your location to get the coordinates')

    with st.form("apartment_form"):
        latitude = st.number_input("Latitude", format="%.5f", step=0.000001)
        longitude = st.number_input("Longitude", format="%.5f", step=0.000001)
        stratum = st.number_input("Stratum", min_value=1, max_value=6, step=1)
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
        constructed_area = st.number_input("Constructed Area (m²)", min_value=10, step=1)
        house_age = st.number_input("House Age (years)", min_value=0, step=1)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
        parking = st.number_input("Number of Parking Spots", min_value=0, step=1)
        administration = st.number_input("Administration Fee (COP)", min_value=0, step=50000)
        floor = st.number_input("Floor", min_value=1, step=1)
        submitted = st.form_submit_button("Submit")

    new_data = None

    if submitted:
        if 4.45 <= latitude <= 4.83 and -74.25 <= longitude <= -73.8:
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
            new_data = pd.DataFrame([new_data])
            st.write("### 📋 Entered Data:")
            st.dataframe(new_data)

            try:
                prediction = loaded_model.predict(new_data)
                st.success(f"The predicted price is: {prediction[0]:,.0f} COP")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.write("##The specified location is outside Bogotá. Please make sure the latitude is within the range [4.45, 4.83] and the longitude [-74.25, -73.8].")

with tab2:
    st.title('Exploratory Analysis of the rent market')
    st.subheader('Rental Price by District')

    st.write('The next figure shows the distribution of rental prices by district in Bogotá, illustrating how the value increases the further \
             north-east the district is located, with Chapinero an Usaquen leading the list and Bosa being the cheapest one')
    sns.set_style('darkgrid')
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor('#F0F0F0') 
    ax1.set_facecolor('#EAEAEA')
    mean_price_localidad = apto_data.groupby('Localidad')['rent_price'].mean().sort_values(ascending=False)
    bar_color = '#778899'
    sns.barplot(
        x=mean_price_localidad.index,
        y=mean_price_localidad.values,
        ax=ax1,
        color=bar_color
    )

    ax1.set_ylabel('Mean Rent Price (COP)')
    ax1.set_title('Mean Rent Price by District')
    sns.despine(left=True, bottom=True)
    plt.setp(
        ax1.get_xticklabels(), 
        rotation=45,          
        ha='right',           
        rotation_mode='anchor'
    )
    plt.tight_layout()

    st.pyplot(fig1)
    #----------
    st.subheader('Mean Constructed Area by District')

    st.write('The following plot shows the mean constructed squared meters in apartments by distric. As the data illustrates, Chapinero and Usaquen are once again leading the list.')

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#F0F0F0') 
    ax2.set_facecolor('#EAEAEA')
    mean_constructed_area_district = apto_data.groupby('Localidad')['constructed_area'].mean().sort_values(ascending=False)
    bar_color = '#778899'
    sns.barplot(
        x=mean_constructed_area_district.index,
        y=mean_constructed_area_district.values,
        ax=ax2,
        color=bar_color
    )

    ax2.set_ylabel('Mean constructed area')
    ax2.set_title('Mean Constructed Area by District')
    sns.despine(left=True, bottom=True)
    plt.setp(
        ax2.get_xticklabels(), 
        rotation=45,          
        ha='right',           
        rotation_mode='anchor'
    )
    plt.tight_layout()

    st.pyplot(fig2)
    #-------
    st.subheader('Mean house age by District')

    st.write('This plot illustrates the mean house age of properties by district. Once again Chapinero and Usaquén lead the ranking, with Antonio Nariño joining them in the top three')

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor('#F0F0F0') 
    ax3.set_facecolor('#EAEAEA')
    mean_house_age_district = apto_data.groupby('Localidad')['house_age'].mean().sort_values(ascending=False)
    bar_color = '#778899'
    sns.barplot(
        x=mean_house_age_district.index,
        y=mean_house_age_district.values,
        ax=ax3,
        color=bar_color
    )

    ax3.set_ylabel('Mean House Age (years)')
    ax3.set_title('Mean House Age by District')
    sns.despine(left=True, bottom=True)
    plt.setp(
        ax3.get_xticklabels(), 
        rotation=45,          
        ha='right',           
        rotation_mode='anchor'
    )
    plt.tight_layout()

    st.pyplot(fig3)

    #-------
    st.subheader('Rental Cost per square meter by District')

    st.write('')

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.patch.set_facecolor('#F0F0F0') 
    ax4.set_facecolor('#EAEAEA')

    data['rent_per_m2'] = data['rent_price']/data['constructed_area']
    renta_cost_square_meter = data.groupby('Localidad')['rent_per_m2'].mean().sort_values(ascending=False)
    print(renta_cost_square_meter.values)
    bar_color = '#778899'
    sns.barplot(
        x=renta_cost_square_meter.index,
        y=renta_cost_square_meter.values,
        ax=ax4,
        color=bar_color
    )

    ax4.set_ylabel('Rent price per square meter')
    ax4.set_title('Rent price per square meter by District')
    sns.despine(left=True, bottom=True)
    plt.setp(
        ax4.get_xticklabels(), 
        rotation=45,          
        ha='right',           
        rotation_mode='anchor'
    )
    plt.tight_layout()

    st.pyplot(fig4)