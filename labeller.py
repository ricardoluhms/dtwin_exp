import os
import xml.etree.ElementTree as ET
from binner import Bin_2_Array
from outter import Reader
import cv2
import numpy as np
from tqdm import tqdm


def normalize_img(array):
    return (255*(array/(array.max()-array.min()))).astype("uint8")

class Out_reader(Reader):

    def __init__ (self,folder_path):
        self.path_list=[]
        self.folder_path=folder_path
        valid_files = ".out"
        for file in os.listdir(self.folder_path):
            if (file.endswith(valid_files)and file.split('.')[0]=="amplitude"):
                old=self.folder_path+'/'+file
                new=self.folder_path+'/0'+file
                os.rename(old, new)
            if file.endswith(valid_files):
                n_path=os.path.join(self.folder_path,file)
                if file=='0amplitude.out':
                    self.amplitude_reader = Reader(n_path)
                elif file=='ambient.out':
                    self.ambient_reader = Reader(n_path)
                elif file=='flags.out':
                    self.flags_reader = Reader(n_path)
                elif file=='phase.out':
                    self.phase_reader = Reader(n_path)
                if n_path not in self.path_list:
                    self.path_list.append(n_path)
        print("files in the selected folder: ")
        print(self.folder_path)

    def out_2_array(self,file_type="Amplitude"):
        ret=True
        if file_type=="Amplitude" or file_type==0:
            core=self.amplitude_reader
        elif file_type=="Ambient" or file_type==1:
            core=self.ambient_reader
        elif file_type=="Flags" or file_type==2:
            core=self.flags_reader
        elif file_type=="Phase" or file_type==3:
            core=self.phase_reader 
        total=core.frames_count
        if total >500:
            total=500
        for count in tqdm(range(total)):
            ret, frame = core.read()
            frame=np.array([[frame.reshape((core.height,core.width))]])
            if count==0:
                reshaped_group=frame
            elif count>0 and ret:
                reshaped_group=np.vstack((reshaped_group,frame))
            #print(reshaped_group.shape)
        return reshaped_group
            

    def out_2_array_complete(self):
        for num in range(len(self.path_list)):
            print("Reading Frames from: ", self.path_list[num])
            if num==0:
                reshaped_group=self.out_2_array(file_type=num)
            else:
                reshaped_group=np.hstack((reshaped_group,self.out_2_array(file_type=num)))
        return reshaped_group

class Cropper():

    @staticmethod
    def add_label(label_num=1, std_label=True):
        if std_label==False:
            label=input("Set Label Name for cropped region:")
        else:
            label="obj_label-"+str(label_num)
        return label

    @staticmethod
    def std_window_show(window_name,array):
        cv2.namedWindow(window_name,cv2.WINDOW_KEEPRATIO)
        cv2.imshow(window_name,array)

    @staticmethod
    def _crop_coord_detect(refPt,img):
        x_max=max(refPt[0][0],refPt[1][0])
        x_min=min(refPt[0][0],refPt[1][0])
        y_max=max(refPt[0][1],refPt[1][1])
        y_min=min(refPt[0][1],refPt[1][1])
        roi = img[y_min:y_max, x_min:x_max]
        roi_coord = [x_min,y_min,x_max,y_max]
        return roi, roi_coord

    def click_and_crop(self,event, x, y, flags, param):
        # grab references to the global variables
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        global refPt, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True  # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that the cropping operation is finished
            refPt.append((x, y))
            cropping = False
        if len(refPt)==2:
            cv2.rectangle(img, refPt[0], refPt[1], (0,255,255), 1)
            roi, _=self._crop_coord_detect(refPt,img)
            self.std_window_show('Crop',roi)
	
    def crop_img(self,img_in,img_name ,crop_obj=1, std_label=True):
        global img, refPt, cropping
        refPt=[];
        cropping = False
        img = img_in.copy()
        img_clone = img.copy()
        roi_img = img.copy()
        window_name=("Image:" + str(img_name))

        cv2.namedWindow(window_name,cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(window_name, self.click_and_crop)
        while True:
            skip_crop=False
            cv2.imshow(window_name, img)
            key = cv2.waitKey(1) & 0xFF   
            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                print("r key pressed")
                img = img_clone.copy()
            # if the 'c' key is pressed, break from the loop
            elif key == ord("l") or key == ord("l") or key==13:
                if len(refPt) < 2:
                    skip_crop=True
                else:
                    cv2.destroyWindow(window_name)
                    break

        if (len(refPt) == 2 and skip_crop==False):
            _,roi_coord = self._crop_coord_detect(refPt,img_clone)
            ## roi_coord format output= [x_min,y_min,x_max,y_max]
        cv2.destroyAllWindows()
        crop_label=self.add_label(label_num=crop_obj, std_label=True )
        color=(255,255,255)
        cv2.putText(roi_img, crop_label, (roi_coord[0], roi_coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, lineType=cv2.LINE_AA)
        roi_img=cv2.rectangle(roi_img, (roi_coord[0], roi_coord[1]), (roi_coord[2], roi_coord[3]), color, 1)
        return roi_coord, crop_label, roi_img

class XMLer():

    def __init__(self,img_file_path,annotate_user="Luhm",xml_file_path="output.xml"):
        self.img_file_path=img_file_path
        self.xml_file_path=xml_file_path
        self.root=ET.Element("Annotation")
        ### folder value should be "DATASET_NAME" in the following example:
        # "c:/DATASET_FOLDER_NAME/JPEGImages/image.jpeg"
        img_path=img_file_path.split("/")

        img_folder=ET.SubElement(self.root,"folder").text=img_path[len(img_path)-3]
        img_file=ET.SubElement(self.root,"filename").text=img_path[len(img_path)-1]
        source=ET.SubElement(self.root,"source")
        ET.SubElement(source,"database").text="Moving Objects Experiment Aug.2020 PUCPR"
        ET.SubElement(source,"Annotation User").text=annotate_user
        self.img=cv2.imread(img_file_path); 
        height, width, depth = self.img.shape
        size=ET.SubElement(self.root,"size")
        ET.SubElement(size,"width").text=str(width)
        ET.SubElement(size,"height").text=str(height)
        ET.SubElement(size,"depth").text=str(depth)
        self.tree=ET.ElementTree(self.root)
        self.tree.write(xml_file_path)
    
    def add_object(self,obj_name,coord=[]):
        obj=ET.SubElement(self.root, "object")
        ET.SubElement(obj,"name").text=str(obj_name)
        bbox=ET.SubElement(obj,"bndbox")
        ET.SubElement(bbox,"xmin").text=str(coord[0]) #xmin
        ET.SubElement(bbox,"ymin").text=str(coord[1]) #ymin
        ET.SubElement(bbox,"xmax").text=str(coord[2]) #xmax
        ET.SubElement(bbox,"ymax").text=str(coord[3]) #ymax
        self.tree=ET.ElementTree(self.root)
        self.tree.write(self.xml_file_path)

class Annotate():

    def __init__(self,video_fld,image_output_fld,xml_annot_fld,user_name="Luhm",video_type="out"):
        if video_type=="out":
            reader = Out_reader(video_fld)
            self.video_array = reader.out_2_array_complete()

        elif video_type=="bin":
            self.video_array=Bin_2_Array(video_fld).reshaped_grouped() # video_array  [frames,fmap,240,320]
        self.video_in=video_fld
        self.iof=image_output_fld
        self.xaf=xml_annot_fld
        self.user=user_name
    
    def one_frame(self, frame_n=100,    file_type="Amplitude", 
                        objects=1,  file_name="Image_",object_names=["object_1","object_2","object_3"]):
        # 0 is "Amplitude"; 1 is "Ambient"; 3 is "Depth"; 4 is "Phase"
        types={"Amplitude":0, "Ambient":1, "Depth":2, "Phase":3}
        
        frame=self.video_array[frame_n,types[file_type],:,:].copy()
        print("start",frame.max(),frame.min(),frame.mean())
        if file_type=="Amplitude":
            
            mask=frame>400
            frame[mask]=450
        frame=normalize_img(frame)
        print("end",frame.max(),frame.min(),frame.mean())
        #from IPython import embed; embed()

        img_path=self.iof+"/"+file_name
        count=0
        
        i_fname = self.iof+"/"+file_name+str(count)+".jpg"
        x_fname = self.xaf+"/"+file_name+str(count)+".xml"
        while os.path.isfile(i_fname):
            count += 1
            i_fname=self.iof+"/"+file_name+str(count)+".jpg"
            x_fname=self.xaf+"/"+file_name+str(count)+".xml"
        else:
            crp=Cropper()
            for crp_num in range(objects):
                if crp_num==0:
                    cv2.imwrite(i_fname, frame) 
                    xm=XMLer(i_fname,annotate_user=self.user,xml_file_path=x_fname)
                    roi_coord, crop_label, _ = crp.crop_img(frame,i_fname,crop_obj=crp_num)
                    xm.add_object(crop_label,coord=roi_coord)
                    #from IPython import embed; embed()
                else:
                    roi_coord, crop_label, _ = crp.crop_img(frame,i_fname ,crop_obj=crp_num)
                    xm.add_object(crop_label,coord=roi_coord)
            

    def all_frames(self, file_type="Amplitude", 
                        objects=2,  file_name="Image_"):
        for frame_n in range(self.video_array.shape[0]):
            if frame_n==0:
                pass
            else:
                self.one_frame(frame_n=frame_n,    file_type=file_type, 
                            objects=objects,  file_name=file_name)


if __name__ == '__main__':
    print("Running Labeller")
    xml_annot_fld="D:/media/ssd/ssd_data/Experimentos//out_files_input/1 captura/Annotation"
    input_fld="D:/media/ssd/ssd_data/Experimentos/out_files_input/1 captura"
    image_output_fld="D:/media/ssd/ssd_data/Experimentos//out_files_input/1 captura/JPEGImages"
    an=Annotate(input_fld,image_output_fld,xml_annot_fld,user_name="Luhm")
    #an.one_frame()
    an.all_frames(objects=2)
    #from IPython import embed; embed()


