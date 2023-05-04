import os
import streamlit as st # for the GUI
import imageio as iio # for tiff handling
import retinalseg as rs
from datetime import datetime
import pandas as pd # for handling data
import shadowRemoval as sr #for shadow removal
import cv2
from tqdm import tqdm
# from register import *

"""
Troubles in allocating enough resources to run the script successfully on the GPU.
Hence, the code below ensures that the code runs on the CPU.
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 


# Layout of the app
st.header("Reinal Image Segementation of OCT data")
# st.write("Choose any image and get corresponding binary mask:")
img_stk = st.file_uploader("Upload image stack... (in .tiff format)")

if(img_stk != None):
    img_name = img_stk.name
    img_stk = iio.mimread(img_stk)
    
    st.subheader("Stack Preprocessing (Image Registration and Shadow Removal)")
    # flag=True
    # # Image Registration
    # if st.checkbox('Register Images') and flag==True:
    #     print('----------------------started registration --------------------')
    #     arr = st.text_input('Upload indexes of reference, starting and ending images as ref1,start1,end1,ref2,start2,end2').split(',')
    #     idxs = []
    #     if arr:
    #         for i in range(0, len(arr), 3):
    #             idxs.append([int(arr[i]), int(arr[i+1]), int(arr[i+2])])

    #         img_stk = registration(idxs, json_dir='labels_my-project-name_2022-11-09-02-04-32.json', img_stk = img_stk)
            # img_stk=[]
            # for batch_idx, sample in enumerate(registered_imgs):
            #     img_stk.append(sample['img'].numpy())
        
        # for i in img_stk:
        #     print(i.shape)
        # flag=False
        # print('----------------------ended registration------------------------')
        
    
    # Shadow Removal
    if(st.button("Remove shadows")):
        """
        Shadow Removal pipeline
        """
        shad = sr.load_model()
        print('------------------started shadow removal-------------------')
        print("---preprocessing started---")
        img_stk = [sr.preprocess(img) for img in tqdm(img_stk)]
        print("---removal started---")
        img_stk = sr.remove(shad, img_stk)

        now = datetime.now()
        time_str = now.strftime("%d%m%Y_%Hh%Mm%Ss")
        iio.mimwrite("Final app/saved_results/"+time_str+"_"+img_name[:-5]+"_sr.tiff", img_stk)

        print('------------------ended shadow removal---------------------')

    st.subheader("Retinal Layers Prediction")
    rs_model_path = st.text_input(label="Enter model name from saved_models directory to use a pre-trained model")
    if(st.checkbox('Enable Training')):
        mask_stk = st.file_uploader("Upload corresponding mask stack...")
        if(mask_stk != None):

            mask_stk = iio.mimread(mask_stk)
            epochs = st.number_input(label="Enter the number of epochs", min_value=2)

            if(st.button('Train')):
                print('------------------started rs_training---------------------')
                if(rs_model_path == ""):
                    rs_model_path = None
                rs_model = rs.get_model(rs_model_path) # initialize retinal segmentation model 

                score, loss, pred = rs.train(img_stk, mask_stk, rs_model, epochs)

                now = datetime.now()
                time_str = now.strftime("%d%m%Y_%Hh%Mm%Ss")
                iio.mimwrite("Final app/saved_results/"+time_str+"_"+img_name[:-5]+"_rs.tiff", pred)

                score_data = pd.DataFrame(score, columns=["Train IOU score", "Validation IOU score"])
                st.line_chart(score_data)

                loss_data = pd.DataFrame(loss, columns=["Train loss", "Validation loss"])
                st.line_chart(loss_data)
                print('------------------ended rs_training---------------------')

    else:
        if(st.button('Predict')):
            # path = "H:/Pranjal/Final app/saved_models"
            # for i in os.listdir(path):
            #     rs_model = rs.get_model(i) # initialize retinal segmentation model 
            #     pred = rs.test(img_stk, rs_model)

            #     now = datetime.now()
            #     time_str = now.strftime("%d%m%Y_%Hh%Mm%Ss")
            #     iio.mimwrite("Final app/saved_results/"+time_str+"_"+img_name[:-5]+"_rs.tiff", pred)
            
            if(rs_model_path == ""):
                    rs_model_path = None
            
            print('------------------started rs_prediction---------------------')

            rs_model = rs.get_model(rs_model_path) # initialize retinal segmentation model 
            pred = rs.test(img_stk, rs_model)

            now = datetime.now()
            time_str = now.strftime("%d%m%Y_%Hh%Mm%Ss")
            iio.mimwrite("Final app/saved_results/"+time_str+"_"+img_name[:-5]+"_rs.tiff", pred)
            
            print('------------------ended rs_prediction---------------------')