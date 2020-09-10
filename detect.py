
from torchvision import transforms
from utilsv2 import *
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import cv2
import numpy as np
import os
from tof2image import Bin_2_Array
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = "D:/media/ssd/ssd_data/Experimentos/outputs/checkpoint_custom_ssd300v2.pth.tar"
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

labels=('cubo_gran', 'cubo_peq', 'piramide_gran', 'piramide_peq', 'esfera')
label_map = {k: v + 1 for v, k in enumerate(labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    
    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("./calibril.ttf", 15)
    
    #from IPython import embed; embed()
    det_scores=det_scores[0].to('cpu').detach().tolist()
    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])

        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
             det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        score=str(round(det_scores[i]*100,3))+" % "

        if det_scores[i]<0.5:
            label="Unknown:" + score
            
        else:
            label=det_labels[i].upper()+":" + score
        

        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        #from IPython import embed; embed()
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])

        draw.text(xy=text_location, text=label, fill='black',
                  font=font)
    del draw    

    return annotated_image

def video_detect(video_filepath,output_mode= "video"):
    # Get folder path and file_name
    split_me="/"
    ls=video_filepath.split(split_me)
    path=split_me.join(ls[:len(ls)-1])
    file_name=ls[len(ls)-1].split(".")[0]

    #Open Video
    video =cv2.VideoCapture(video_filepath)

    #Open Video Codec
    fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #fourcc=cv2.VideoWriter_fourcc(*'MP4V')

    #Get Video Properties (Total Frames, FPS)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    tt_frames=video.get(cv2.CAP_PROP_FRAME_COUNT)

    #Open first Frame and convert from OpenCV to PIL
    success,image = video.read()
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #Get Imag Shape
    shape =image.size

    #Start Video Output
    #from IPython import embed; embed()
    output_path = path+"/"+str(file_name)+'_output.avi'
    out = cv2.VideoWriter(output_path, fourcc, fps, shape)

    #Run SSD detection
    detection=detect(image, min_score=0.30, max_overlap=0.25, top_k=200)
    index=0
    #from IPython import embed; embed()
    while success:
        if output_mode=="video":
            out.write(np.array(detection))
        elif output_mode=="pictures":
            image_path=path+"/"+"frame"+str(index)+".jpg"
            cv2.imwrite(image_path, np.array(detection))   
        success,image = video.read()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detection=detect(image, min_score=0.30, max_overlap=0.25, top_k=200)
        index += 1

def frame2video_detect(folders,video_output,fps=30,shape=(320,240)):
    for folder in folders:
        img_folder=folder+"/JPEGImages"
        exp_name=folder.split("/")[-1]
        output_path=video_output+"/"+str(exp_name)+'_output2.avi' #_output2.avi'
        fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #'m', 'p', '4', 'v'
        #from IPython import embed; embed()
        out = cv2.VideoWriter(output_path, fourcc, fps, shape)

        for image_file in  os.listdir(img_folder):
            img_path =img_folder+"/"+image_file
            original_image = Image.open(img_path, mode='r')
            original_image = original_image.convert('RGB')
            annotated = detect(original_image, min_score=0.30, max_overlap=0.2, top_k=200)
            out.write(np.array(annotated))

        out.release()

def binfile2video_detect(bin_input_fld,video_output,start_frame=0,max_frames=1000,frame_skip=1,
                        file_type="Amplitude",fps=30,shape=(320,240)):
    video_buffer=Bin_2_Array(bin_input_fld).reshaped_grouped()
    exp_name=bin_input_fld.split("/")[-1]
    types={"Amplitude":0, "Ambient":1, "Depth":2, "Phase":3}
    output_path=video_output+"/"+str(exp_name)+"-"+str(file_type)+'_output_complete.avi' #_output2.avi'
    fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    out = cv2.VideoWriter(output_path, fourcc, fps, shape)
    inference_perf=[]
    for num, fs in enumerate(range(start_frame,max_frames,frame_skip)):
        frame=np.array(video_buffer[fs,types[file_type],:,:])

        if file_type=="Amplitude":
            mask=frame>400
            frame[mask]=450
        #from IPython import embed; embed()
        image = Image.fromarray(frame)
        image=image.convert('RGB')
        start_time=time()
        annotated = detect(image, min_score=0.30, max_overlap=0.25, top_k=200)
        end_time=time()
        inference_perf.append( round ( 1/ (end_time-start_time) ) )
        out.write(np.array(annotated))
    inference_perf_arr=np.array(inference_perf)
    print("Max Inference:", inference_perf_arr.max(), "FPS")
    print("Avg Inference:", inference_perf_arr.mean(), "FPS")
    print("Min Inference:", inference_perf_arr.min(), "FPS")
    out.release()



if __name__ == '__main__':
    folders = [ "D:/media/ssd/ssd_data/Experimentos/bin_exp1",      
                "D:/media/ssd/ssd_data/Experimentos/bin_exp2",
                "D:/media/ssd/ssd_data/Experimentos/bin_exp3"]
    #"D:/media/ssd/ssd_data/Experimentos/bin_exp1",
    video_output="D:/media/ssd/ssd_data/Experimentos/outputs/videos"

    for bin_input_fld in folders:

        binfile2video_detect(bin_input_fld,video_output,start_frame=0,max_frames=1000,frame_skip=1,
                            file_type="Amplitude",fps=30,shape=(320,240))
        #from IPython import embed;embed()
        # video_path = "D:/media/ssd/ssd_data/Teste/dog_cacau.mp4"
        # video_detect(video_path,output_mode= "video")
