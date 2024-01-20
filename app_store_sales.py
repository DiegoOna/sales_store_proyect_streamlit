# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:00:23 2024

@author: User
"""

import streamlit as st
import joblib
import numpy as np 
from PIL import Image
import pandas as pd
import sklearn

st.set_page_config(page_title='Rossmann supermarkets',
                   page_icon='🌎',
                   layout='centered',
                   initial_sidebar_state='auto')



#st.title('PET PLANET')
image = Image.open('Rossman_528_299.png')
#nuevo_tamano_img=(669,155)
nuevo_tamano_img=(528,250)
image_redimensionada=image.resize(nuevo_tamano_img)
st.image(image_redimensionada, caption='by Ing. Diego Oña ')
#st.subheader("Encuentra todo al mejor precio.")
#st.image('MarcaEMPRESA ML-JPG.jpg',width=100)

#=========================================================================
#=========================================================================


# Crear una barra lateral
sidebar = st.sidebar

# Añadir un título a la barra lateral
sidebar.title("ROSSMANN SUPERMARKETS ")
sidebar.header("PRONÓSTICO DE VENTAS ")
st.sidebar.image('MarcaPersonal_ML_JPG.jpg',width=100)
#===============================================================
#                  SECCIÓN BARRA LATERAL
#===============================================================
sidebar = st.sidebar
#st.image('MarcaEMPRESA ML-JPG.jpg',width=100)
# Añadir un cuadro de selección a la barra lateral

clientes_custom = sidebar.text_input(label='Cantidad de Clientes',value=1)
try:
    customers=int(clientes_custom)
    if customers<=0:
       st.error('CAMPO: CANTIDAD DE CLIENTES "¡DETECTA NÚMERO NEGATIVO !"', icon="🚨")
       sidebar.info(f'Ingresar un número positivo en el campo para continuar con el resto de requerimientos',icon="ℹ️")  
except ValueError: 
       st.error('CAMPO: CANTIDAD DE CLIENTES "¡DETECTA CARÁCTER NO NUMÉRICO!"', icon="🚨")
       sidebar.info(f'Ingresar un número positivo en el campo para continuar con el resto de requerimientos..',icon="ℹ️")
                
                  
compet_open = sidebar.text_input("Distancia de la competencia más cercana (m)", value=0)
sidebar.info(f'PROMOCIÓN 1 (Opcional): Se aplica por un día' ) 


promo_1= sidebar.selectbox("Aplica promoción 1", ['Seleccionar...',"Yes", "No"])
if promo_1=='Yes':
        promo1=1    
elif promo_1=='No':
        promo1=0            

    
    
try:
                   competition_open=float(compet_open)
                   if competition_open<0:
                      st.error('CAMPO: DISTANCIA DE LA COMPETENCIA...! "INGRESAR NÚMERO POSITIVO" ', icon="🚨")
                   else:
                       if sidebar.button("<< APLICAR >>"):
                                          #Data para predecir,toma todos los valores de ingreso y se aplica la predicion ene l modleo 
                                          model_filename = 'sales_store.pkl'
                                          # Cargamos el modelo desde el archivo
                                          loaded_model = joblib.load(model_filename)
                                          print("Hemos Cargado el modelo...")
                                          try:
                                             new_data={'Customers':[customers] ,
                                                       'Promo':[promo1],
                                                       'CompetitionDistance':[competition_open]
                                                       }
                                          
                                             #Esta data se últiliza para hacer una predición y comparación cuando la competencia esta junto es decir CompetitionOpen=0
                                             data_compare={'Customers':[customers] ,
                                                           'Promo':[promo1],
                                                           'CompetitionDistance':[0]
                                                           }
                                             #Data para la presentación. Esta data se muestra solo en la interface de streamlit, con los valores que se han ingresado
                                             data_visualize={'Clientes':[customers] ,
                                                     'Promoción 1':[promo_1],
                                                     'Distancia Competencia (m)':[competition_open]
                                                     }
                                             new_data=pd.DataFrame(new_data)
                                             data_visualize=pd.DataFrame(data_visualize)
                                             data_compare=pd.DataFrame(data_compare)
                                             st.header('📄 Datos ingresados: ')
                                             st.write(data_visualize)
                                             prediction= loaded_model.predict(new_data)
                                             prediction_base= loaded_model.predict(data_compare)

                                             print('Ejecutando predicción...')
                                             st.success(f' VENTAS PREDICCIÓN:  {abs(prediction)} USD', icon="🏆")
                                          
                                             # Calcular el delta_ventas como un porcentaje
                                             porcentaje_delta_ventas=(prediction[0]/prediction_base[0])*100
                                             # Muestra las métricas en Streamlit
                                             st.metric(label="📈Ventas Predicción", value=f"{abs(prediction[0]):.2f}", delta=f"{porcentaje_delta_ventas:.2f}%")

                                             col1, col2 = st.columns(2)
                                             col1.metric("👩‍👩‍👦‍👦 Customers", f"{customers}", "")
                                             col2.metric("🏫 CompetitionOpen", f"{competition_open:.2f} m", "")

                                             # Muestra información adicional si hay un cambio en las ventas
                                             if porcentaje_delta_ventas != 0:
                                               st.info(f'Variación de ventas: {round(porcentaje_delta_ventas, 2)} % cuando la competencia más cercana está a {competition_open} metros.', icon="👉")    
                                          except NameError:                                             
                                                     st.error('CAMPO: Promoción 1 ¡Seleccionar..!',icon="🚨")
                       else:
                            st.info(f'EL PRONÓSTICO DE VENTAS SE MOSTRARÁ AQUÍ',icon="👉") 
except ValueError: 
    st.error('CAMPO: DISTANCIA DE LA COMPETENCIA...!solo permite ingresar números', icon="🚨")

 
               
                                     