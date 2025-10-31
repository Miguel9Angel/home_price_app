import streamlit as st 
import pandas as pd
import pydeck as pdk
import requests
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import plot_importance

st.set_page_config(
    page_title="Bogotá Rent Price Prediction",  
    page_icon="🏙️",                            
    layout="wide",                             
    initial_sidebar_state="expanded"     
)

model_filename = '../model/best_xgb_pipeline.pkl'

@st.cache_resource
def load_model(): 
    try:
        paths_to_try = [
            '../model/best_xgb_pipeline.pkl', 
            './model/best_xgb_pipeline.pkl'   
        ]
        
        loaded_model = None
        for path in paths_to_try:
            try:
                loaded_model = joblib.load(path)
                print(f"Model '{path}' loaded successfully.")
                break
            except FileNotFoundError:
                continue 
        
        if loaded_model is None:
            raise FileNotFoundError(f"Model not found in any path: {paths_to_try}")

        return loaded_model
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

loaded_model = load_model()
if loaded_model is None:
    st.error("🚨 File wasn't found")
    st.stop()
@st.cache_data
def load_data():
    try:
        return pd.read_csv('../data/apartments_bogota.csv') 
    except FileNotFoundError:
        return pd.read_csv('./data/apartments_bogota.csv')

data = load_data()
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

    st.write('The rent price per square meter keep Chapinero as one of the most expensives in the city, with others districs like Santa Fe and Candelaria which are the city center, and keeping Bosa as the cheapest distric to live.')

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.patch.set_facecolor('#F0F0F0') 
    ax4.set_facecolor('#EAEAEA')

    data_clean = data[data['constructed_area']>0].copy()
    data_clean['rent_per_m2'] = data_clean['rent_price']/data_clean['constructed_area']
    renta_cost_square_meter = data_clean.groupby('Localidad')['rent_per_m2'].mean().sort_values(ascending=False)
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
    
    #-----BoxPlot stratum
    st.subheader('Rental Price y Stratum')
    st.write("The data exhibits a strong , positive correlation between the socio-economic stratum and the rental price, confirming the expected market dynamics")
    
    sns.set_style('darkgrid')
    
    fig5, ax5 = plt.subplots(figsize=(10,6))
    sns.boxplot(
        x = 'stratum',
        y='rent_price',
        data=apto_data,
        ax=ax5,
        palette='Blues_d',
        medianprops={'color':'darkred', 'linewidth':2}
    )
    
    ax5.set_title('Rental price distribution by stratum', fontsize=16)
    ax5.set_xlabel('Stratum')
    ax5.set_ylabel('Rental Price (COP)')
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    st.pyplot(fig5)
    
with tab3: # Suponiendo que esta es la pestaña 3
    st.header("Methodology and Model Evaluation")
    
    st.subheader("Origin and Preparation of Data")
    st.markdown("""
        All the data was obtained from the FincaRaiz website using the BeautifulSoup library to extract the required features, resulting in a total of 8,562 records collected.
        The steps of the cleaning process were:
        * All the outliers were handled with different tehcniques depending on the variable context. Check my Github for more details.
        * Finca Raiz website provides information about the apartment facilities available. However, after performing a mutual information regression analysis between the input
        and output variables, these facilities were found to hace less than 1% predictive power.
    """)
    
    st.subheader("Model Predictor 🧠")
    st.markdown("""
        Four of the basic Machine learning models were used LinearRegression, LassoCV, RandomForest and XGBoost these last two models were the better ones with the lowest RMSE value.
    """)
    data_models = {
        'Model': ['XGBoost', 'RandomForest', 'LinearRegression', 'LassoCV'],
        'RMSE Mean': [1.712468e+06, 1.724811e+06, 2.066443e+06, 2.066493e+06],
        'RSME Std': [95403.742644, 102536.007085, 97564.781750, 97411.108760]
    }
    df_models = pd.DataFrame(data_models)
    df_models = df_models.sort_values(by='RMSE Mean', ascending=True).reset_index(drop=True)
    st.dataframe(df_models,
                hide_index=True,
                use_container_width=True)
    
    st.info(f"""
        **The model with the best results is XGBoost (Test Set):**
        * **RMSE Mean:** **$1.712.468 COP**
    """)
    st.markdown("""
        The mean error of the model's predictio is approximately 1.712.468 COP. This is mainly due to some apartments with very high rental prices.
        Apartments with rents above 8.000.0000 COP show significant variability in the prediction process, whereas the model performs much
        better for lower-priced apartments.
    """)
    
    st.subheader("Features Importance")
    st.write('''
        The next plot shows the importance score of each feature in the model training procces
    ''')
    
    fig6, ax6 = plt.subplots(figsize=(10,6))
    xgb_model = loaded_model.named_steps['regressor']
   
    plot_importance(
        xgb_model, 
        ax=ax6,
        max_num_features=10, 
        height=0.5,
        show_values=False 
    )
    plt.title('Features importance (XGBoost)')
    plt.tight_layout()
    st.pyplot(fig6)

    feature_data = {
    "Feature": [
        "f0", "f1", "f2", "f3", "f4", 
        "f5", "f6", "f7", "f8", "f9"
    ],
    "Description": [
        "longitude", "latitude", "constructed_area", "house_age",
        "administration", "floor", "stratum", "bathrooms",
        "bedrooms", "parking"
    ]
    }

    features_df = pd.DataFrame(feature_data)

    st.subheader("Feature Index Mapping")
    st.write('This shows that location is the most important feature for determining the rental price of apartment.' \
                'Next, the constructed area can be seen as one of the most relevant features, while parking is the least important of all.')
    st.dataframe(features_df, use_container_width=True)

    st.markdown("---")
    st.subheader("🔗 Project Repository")

    st.markdown("""
    For more details about the data collection process, preprocessing pipeline, and model training code,  
    visit my GitHub repository:

    👉 [**Miguel Soler — Home Price Prediction Bogota (GitHub)**](https://github.com/Miguel9Angel/home_price_bogota)
    """)
