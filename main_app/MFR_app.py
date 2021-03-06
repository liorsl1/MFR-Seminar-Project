from utils import *
import streamlit as st
from mtcnn_cv2 import MTCNN
import pandas as pd
from streamlit import cli as stcli

# Global Constants
CLASSES = {3:'with mask',2:'without mask',1:'without mask',0:'without mask'}
# Haar cascade detectors
haarEye_path = r"Models\HaarCascade\haarcascade_eye.xml"
haarMouth_path = r"Models\HaarCascade\Mouth.xml"
cascadeEye_detector = cv2.CascadeClassifier(haarEye_path)
cascadeMouth_detect = cv2.CascadeClassifier(haarMouth_path)
# MTCNN face detector
detector = MTCNN(min_face_size=130)
# dlib face landmark detector
landmark_predictor = dlib.shape_predictor("Models/DLIB/shape_predictor_68_face_landmarks.dat")
dlib_detector = dlib.get_frontal_face_detector()
# Loading Custom mask detection model
maskModelDir = 'Models/mask_detection/model.h5'
mask_model = tf.keras.models.load_model(maskModelDir)
mask_model.load_weights('Models/mask_detection/mask_weights.h5')
# Loading SOTA FaceArc Model, for face feature embeddings predictions
FaceArcDir = 'Models/ArcFace/TT_efv2_b0_swish_GDC_arc_emb512_dr0_sgd_l2_5e4_bs512_ms1m_randaug_cutout_bnm09_bne1e4_cos16_batch_float16_E50_arc_sgd_LA_basic_agedb_30_epoch_17_0.977333.h5'
m_FaceArc = tf.keras.models.load_model(FaceArcDir,compile = False)

# Main App
def run_MFR():
    # setting the web page
    st.set_page_config(
    page_title="MFR",
    layout="centered",
    initial_sidebar_state="collapsed"
    )
    st.sidebar.button(label='stop')
    _header = '<p style="text-indent: -4em;font-family:sans-serif; color:#304A5E; font-size: 22px; text-align: center; font-weight: bold;">Seminar Project: Masked Face Recognition</p>'
    st.markdown(_header,unsafe_allow_html=True)
    # Setting the streamlit componenets - by order
    PERSONA_MSG = st.empty()
    FRAME_WINDOW = st.empty()
    CROPPED_WINDOW = st.empty()
    POST_INFO = st.empty()
    NEW_ENTRY = st.empty()
    # initializing the text_input for insert_persona
    if 'NAME' not in st.session_state:
        st.session_state.NAME = ''

    e_startTime = 0
    f_startTime = 0
    mask_sTime = 0
    eyes_detectedTime = 0
    face_detectedTime = 0
    mask_detTime = 0
    rec_posColor = (0,255,0)
    rec_negColor = (255,0,0)
    REC_COLOR = rec_negColor
    mask_msg = CLASSES[2]
    (xf,yf,wf,hf) = (0,0,0,0)
    x_face,y_face = 0,0 # face roi top left position
    landmark=[]
    detected_centroids = []
    is_undetected = False
    PERSONA = None
    faces = []
    try:
        cap = cv2.VideoCapture(0)
        collection = get_dbCollection()
        
        while True:
            ret, img = cap.read()
            dyn_img = dynamicBrightness(img,minimum_brightness=0.4)
            copy_img = dyn_img.copy()
            rgb_img = cv2.cvtColor(dyn_img,cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(rgb_img)
            if len(result) > 0:
                boundingBox = result[0]['box']
                faces = [boundingBox]
            else:
                face_detectedTime = 0
            eyes = cascadeEye_detector.detectMultiScale(rgb_img, 1.2, 6) # 1.2 - copy_img scale reduction, 6 - how many neighbors for each prediction, higher value is less detection yet higher quality.
            for (xf,yf,wf,hf) in faces:
                if face_detectedTime == 0:
                    f_startTime = time.perf_counter()
                cv2.rectangle(copy_img, (xf,yf), (xf+wf+1, yf+hf+1), REC_COLOR, 2)
                face = dyn_img[yf : yf + hf + 1, xf: xf + wf + 1]
                x_face, y_face = xf , yf
                x_centroid = xf + wf/2
                y_centroid = yf + hf/2

        # if we got 2 eyes, draw them and check for steadiness
            if len(eyes) == 2:
                if eyes_detectedTime == 0:
                    e_startTime = time.perf_counter()
                
                for count,(eye_x, eye_y, eye_w, eye_h) in enumerate(eyes):
                    if count == 0:
                        eye_1 = np.array((eye_x, eye_y, eye_w, eye_h))
                    else:
                        eye_2 = np.array((eye_x, eye_y, eye_w, eye_h))
                    cv2.rectangle(copy_img,(eye_x,eye_y), (eye_x+eye_w, eye_y+eye_h),REC_COLOR,1)

                # check if the detected eyes are positioned inside our face's frame (the coordinates are contained in the face range)
                bbox_edge = np.array((xf+wf,yf+hf,wf,hf))
                eyes_in_range =  np.all(eye_1 < bbox_edge) and np.all(eye_2 < bbox_edge) and np.any(np.abs(eye_1[:2] - eye_2[:2]) > [8,8])

                if e_startTime > 0 and f_startTime > 0:
                    # if eyes and face are detected for atleast 1 second (meaning detection is steady), and the eyes are in the face's range, proceed to estimate landmarks.      
                    eyes_detectedTime = time.perf_counter() - e_startTime
                    face_detectedTime = time.perf_counter() - f_startTime
                    if eyes_detectedTime >= 0.4 and face_detectedTime >= 0.65 and eyes_in_range:
                        b_mouth = isMouthVisible(cascadeMouth_detect,face)
                        detected_centroids.append([x_centroid,y_centroid])
                        cv2.putText(copy_img, mask_msg, (xf,yf), cv2.FONT_HERSHEY_SIMPLEX, 0.6, REC_COLOR,2)
                        #get estimation of face landmarks
                        if(not b_mouth):
                            prediction = predictMask(mask_model,face)
                            if prediction == 3:
                                if mask_detTime == 0:
                                    mask_sTime = time.perf_counter()
                                if mask_sTime > 0 :
                                    mask_detTime = time.perf_counter() - mask_sTime

                                REC_COLOR = rec_posColor
                                if mask_detTime >= 0.4 :
                                    mask_detTime = 0
                                    aligned_face, is_aligned = alignFace(dyn_img,eye_1,eye_2,detector)
                                    if is_aligned :
                                        face = aligned_face
                                    landmark = landmark_predictor(face,dlib.rectangle(0,0,face.shape[0],face.shape[1])) # face frame and it's dlib frame
                                    if(landmark.num_parts > 0):
                                    # use the middle point of the nose (dynamically changes in accordance to the mask position on the nose)
                                    # point 29 was found to be the ideal point for cropping the part outside the mask
                                        last_nose_pt = landmark.part(30)
                                        cropped_face = face[ 0 : last_nose_pt.y + 1, 0 : face.shape[1]]
                                        identified = True
                                        cv2.circle(copy_img,(last_nose_pt.x + x_face, last_nose_pt.y + 1 + y_face),5,(0,255,255),-1)
                                        CROPPED_WINDOW.image(cropped_face[:,:,::-1],caption = 'Identified Crop')

                                        embedding = predict_embedding(m_FaceArc,cropped_face)
                                        # I allow for maximum of 2 recognition attempts, after the second one we will add the undetected persona to the DB.
                                        PERSONA,top_idents = recognize_persona(collection,embedding,is_undetected,PERSONA_MSG,NEW_ENTRY)
                                        # need to change here, that after undetection, set is_undetected = false again
                                        if PERSONA is None:
                                            is_undetected = True
                                            persona = None
                                            POST_INFO.empty()
                                        else:
                                            is_undetected = False
                                            if PERSONA != None:
                                                conf_rates = [f"{float(ident[1])*100:.2f}" for ident in top_idents]
                                                recog_msg = f'<p style="text-indent: -4em;font-family:system-ui; color:Green; font-size: 20px; text-align: center; font-weight: bold;">Recognized as {PERSONA} ???</p>'
                                                PERSONA_MSG.markdown(recog_msg, unsafe_allow_html=True)
                                                with POST_INFO.expander(label = 'Additional Info'):
                                                    st.write(pd.DataFrame({'Name': top_idents[:,0],
                                                    'Confidence Rate': conf_rates,
                                                    'Last Updated':top_idents[:,2]
                                                    }))
                                                print(PERSONA)                                       
                            else:
                                mask_detTime = 0
                                REC_COLOR = rec_negColor
                            # update mask msg in case of prediction
                            mask_msg = CLASSES[prediction]
                        else:
                            REC_COLOR = rec_negColor
                            mask_msg = CLASSES[2]
                            mask_detTime = 0
                else:
                    print("h")
            else:
                eyes_detectedTime = 0

            FRAME_WINDOW.image(copy_img[:,:,::-1],caption = 'Live Feedback')

    except Exception as e: 
        traceback.print_exc()
        cap.release()
    cap.release()

# For running the code with the command line
if __name__ == "__main__":
    run_MFR()
