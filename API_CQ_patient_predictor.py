import streamlit as st
import pandas as pad 
import numpy as np
from joblib import dump, load
from PIL import Image
import sklearn
import tensorflow as tf
from tensorflow import keras
import time


def main():
    """ fonction principale de prédiction de conformité des CQ patient Halcyon en IMRT Sein """    
   
    
    st.title('Halcyon IMRT patient specific quality assurance prediction')
    st.write("Please enter the complexity indexes")
    
    post = st.text_input("(in the same format as the exemple below, with SAS10 BA BI) : ", "0.0768 85.4112 4.8067")
    indices = post
    
    image_DL = Image.open('DL_model_img.png') 
        
        
    try:
        ## Préparation des données
        indices = indices.split()
        indices_list = []
        for elm in indices:
            indices_list.append(float(elm))
        #st.write(indices_list)
        test = np.array(indices_list)
        #st.write(test.shape)
        indices = test.reshape(1, -1)
        #st.write(indices.shape)
        #indices_DL_all = indices  
        #st.write(indices)
        StandardScaler = load('StandardScaler_SAS10_BA_BI.joblib')
        indices = StandardScaler.transform(indices)
       # st.write(indices)
        
        indices_finale = []
        for elm in indices[0]:
            indices_finale.append(float(elm))
        indices_finale = np.array(indices_finale)
        #st.write(indices_finale)   
        #st.write(indices_finale.shape)    
        
        def deep_learning_classification(indices):
            
            #df_ML = pad.DataFrame(indices, index = ['1'], columns = ['SAS10', 'BA', 'BI'])
            
            # Deep Learning
            #st.write("ok")
            indices = np.asarray(indices)
            indices = [indices]
            #st.write(indices)
            indices = np.asarray(indices)
            #st.write(indices.shape)
            #proba_tensor=tf.convert_to_tensor(df_ML)
            #st.write(proba_tensor)
            #st.write(proba_tensor.shape)
            DL_model = load('DL_model_five_classes.joblib')
            y_pred_prob_DL = DL_model.predict(indices)
            #st.write(y_pred_prob_DL)
            result_DL = []
            result_DL.append(np.where(y_pred_prob_DL[:,0]>0.535858, 1,0))     #### changer le seuil !!!! et prendre en charge les 5 dimensions
            result_DL.append(np.where(y_pred_prob_DL[:,1]>0.535858, 1,0)) 
            result_DL.append(np.where(y_pred_prob_DL[:,2]>0.535858, 1,0)) 
            result_DL.append(np.where(y_pred_prob_DL[:,3]>0.535858, 1,0)) 
            result_DL.append(np.where(y_pred_prob_DL[:,4]>0.535858, 1,0)) 
            
            
            class_25_25 = result_DL[0]
            if class_25_25 == 1:
                result_class_25_25 = "Conformance QC"
            elif class_25_25 == 0:
                result_class_25_25 = "Non-conformance QC"
            else:
                result_class_25_25 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_3_3 = result_DL[1]
            if class_3_3 == 1:
                result_class_3_3 = "Conformance QC"
            elif class_3_3 == 0:
                result_class_3_3 = "Non-conformance QC"
            else:
                result_class_3_3 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_3_2 = result_DL[2]
            if class_3_2 == 1:
                result_class_3_2 = "Conformance QC"
            elif class_3_2 == 0:
                result_class_3_2 = "Non-conformance QC"            
            else:
                result_class_3_2 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_2_3 = result_DL[3]
            if class_2_3 == 1:
                result_class_2_3 = "Conformance QC"
            elif class_2_3 == 0:
                result_class_2_3 = "Non-conformance QC"            
            else:
                result_class_2_3 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_2_25 = result_DL[4]
            if class_2_25 == 1:
                result_class_2_25 = "Conformance QC"
            elif class_2_25 == 0:
                result_class_2_25 = "Non-conformance QC"
            else:
                result_class_2_25 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
            
            CQ_result = [result_class_25_25, result_class_3_3, result_class_3_2, result_class_2_3, result_class_2_25]
                        
            return CQ_result


        predict_btn = st.button('Predict')
        if predict_btn:
            pred = None
            st.empty()    
            my_bar = st.progress(0)
            for percent_complete in range(100):
                 time.sleep(0.013)
                 my_bar.progress(percent_complete + 1)


            ## deep_hybrid_learning_classification ##
            DL_results = deep_learning_classification(indices_finale)
            st.write('For the Deep Hybrid Learning model : \n')
            if DL_results[1] == "Conformance QC":
                        st.success('Prediction result for QC at 3%/3mm is conformance QC !')
            elif DL_results[1] == "Non-conformance QC":
                        st.warning('Prediction result for QC at 3%/3mm is Non-conformance QC !')

            if DL_results[3] == "Conformance QC":
                        st.success('Prediction result for QC at 2%/3mm is conformance QC !')
            elif DL_results[3] == "Non-conformance QC":
                        st.warning('Prediction result for QC at 2%/3mm is Non-conformance QC !')
                
            if DL_results[0] == "Conformance QC":
                        st.success('Prediction result for QC at 2.5%/2.5mm is conformance QC !')
            elif DL_results[0] == "Non-conformance QC":
                        st.warning('Prediction result for QC at 2.5%/2.5mm is Non-conformance QC !')

            if DL_results[2] == "Conformance QC":
                        st.success('Prediction result for QC at 3%/2mm is conformance QC !')
            elif DL_results[2] == "Non-conformance QC":
                        st.warning('Prediction result for QC at 3%/2mm is Non-conformance QC !')
            
            if DL_results[4] == "Conformance QC":
                        st.success('Prediction result for QC at 2%/2.5mm is conformance QC !')
            elif DL_results[4] == "Non-conformance QC":
                        st.warning('Prediction result for QC at 2%/2.5mm is Non-conformance QC !')

          
            st.write("NB : Non-conformance result means a prediction of a  gamma below 95%")                       
            st.image(image_DL, caption='Deep Learning model architecture and performances')

            
    except Exception as e:
        st.write("Modelisation issue, better call ACD : 57.68 or a.corroyer-dulmont@baclesse.unicancer.fr")
        st.write("Error message : " + str(e))

#####  get the error :

if __name__ == '__main__':
    main()
