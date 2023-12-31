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
    
    post = st.text_input("(in the same format as the exemple below, with SAS10 BA BM) : ", "0.723 0.069 30.629")
    indices = post
    
    image_DL = Image.open('image_DL_Sein.png') 
        
        
    try:
        ## Préparation des données
        indices = indices.split()
        indices_list = []
        for elm in indices:
            indices_list.append(float(elm))

        test = np.array(indices_list)
        indices = test.reshape(1, -1)
        #indices_DL_all = indices  
               
        StandardScaler = load('StandardScaler.joblib')
        indices = StandardScaler.transform(indices)
        
        indices_finale = []
        for elm in indices[0]:
            indices_finale.append(float(elm))
            
        
        def deep_learning_classification(indices):
            
            df_ML = pad.DataFrame(indices_finale, index = ['1'], columns = ['SAS10', 'BA', 'BM'])
            
            # Deep Learning
            proba_tensor=tf.convert_to_tensor(df_ML)
            DL_model_Sein = load('DL_model_Sein.joblib')
            y_pred_prob_DHL = DL_model_Sein.predict(proba_tensor)
            result_DL = np.where(y_pred_prob_DHL[:,1]>0.556942, 1,0)

            class_25_25 = result_DL[0]
            if class_25_25 == 0:
                result_class_25_25 = "Conformance QC"
            elif class_25_25 == 0:
                result_class_25_25 = "Non-conformance QC"
            else:
                result_class_25_25 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_3_3 = result_DL[1]
            if class_3_3 == 0:
                result_class_3_3 = "Conformance QC"
            elif class_3_3 == 0:
                result_class_3_3 = "Non-conformance QC"
            else:
                result_class_3_3 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_3_2 = result_DL[2]
            if class_3_2 == 0:
                result_class_3_2 = "Conformance QC"
            elif class_3_2 == 0:
                result_class_3_2 = "Non-conformance QC"            
            else:
                result_class_3_2 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_2_3 = result_DL[3]
            if class_2_3 == 0:
                result_class_2_3 = "Conformance QC"
            elif class_2_3 == 0:
                result_class_2_3 = "Non-conformance QC"            
            else:
                result_class_2_3 = "Modelisation issue, better call ACD : a.corroyer-dulmont@baclesse.unicancer.fr"
                
            class_2_25 = result_DL[4]
            if class_2_25 == 0:
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
            st.write('For the Deep Hybrid Learning model : \n') 
            if deep_learning_classification(indices_finale)[1] == "Conformance QC":
                        st.success('Prediction result for QC at 3%/3mm is conformance QC !')
            elif deep_learning_classification(indices_finale)[1] == "Non-Conformance QC":
                        st.warning('Prediction result for QC at 3%/3mm is Non-conformance QC !')

            if deep_learning_classification(indices_finale)[3] == "Conformance QC":
                        st.success('Prediction result for QC at 2%/3mm is conformance QC !')
            elif deep_learning_classification(indices_finale)[3] == "Non-Conformance QC":
                        st.warning('Prediction result for QC at 2%/3mm is Non-conformance QC !')
                
            if deep_learning_classification(indices_finale)[0] == "Conformance QC":
                        st.success('Prediction result for QC at 2.5%/2.5mm is conformance QC !')
            elif deep_learning_classification(indices_finale)[0] == "Non-Conformance QC":
                        st.warning('Prediction result for QC at 2.5%/2.5mm is Non-conformance QC !')

            if deep_learning_classification(indices_finale)[2] == "Conformance QC":
                        st.success('Prediction result for QC at 3%/2mm is conformance QC !')
            elif deep_learning_classification(indices_finale)[2] == "Non-Conformance QC":
                        st.warning('Prediction result for QC at 3%/2mm is Non-conformance QC !')
            
            if deep_learning_classification(indices_finale)[4] == "Conformance QC":
                        st.success('Prediction result for QC at 2%/2.5mm is conformance QC !')
            elif deep_learning_classification(indices_finale)[4] == "Non-Conformance QC":
                        st.warning('Prediction result for QC at 2%/2.5mm is Non-conformance QC !')

          
            st.write("NB : Non-conformance result means a prediction of a  gamma below 95%")                       
            st.image(image_DL, caption='ROC curve and confusion matrix for the Deep Learning model')

            
    except Exception as e:
        st.write("Modelisation issue, better call ACD : 57.68 or a.corroyer-dulmont@baclesse.unicancer.fr")
        st.write("Error message : " + str(e))

#####  get the error :

if __name__ == '__main__':
    main()
