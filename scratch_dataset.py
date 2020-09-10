import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tof2image import Bin_2_Array, Out_Reader
import json
import cv2
import numpy as np
from tqdm import tqdm


def normalize_img(array):
    return (255*(array/(array.max()-array.min()))).astype("uint8")

class Out_reader(Out_Reader):

    def __init__ (self,folder_path):
        self.path_list=[]
        self.folder_path=folder_path
        valid_files = ".out"
        for file in os.listdir(self.folder_path):
            if (file.endswith(valid_files)and file.split('.')[0]=="amplitude"):
                old=self.folder_path+'/'+file
                new=self.folder_path+'/0'+file
                os.rename(old, new)
                file='/0'+file
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

    def out_2_array(self,file_type="Amplitude", frame_skip=4, max_frames=1000, start_frame=200):
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
        if total >max_frames:
            total=max_frames
        for count in tqdm(range(start_frame,total,frame_skip)):
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
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
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
            cv2.rectangle(img, refPt[0], refPt[1], (0,0,0), 1)
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
        self.root=ET.Element("annotation")
        ### folder value should be "DATASET_NAME" in the following example:
        # "c:/DATASET_FOLDER_NAME/JPEGImages/image.jpeg"
        img_path=img_file_path.split("/")

        img_folder=ET.SubElement(self.root,"folder").text=img_path[len(img_path)-3]
        img_file=ET.SubElement(self.root,"filename").text=img_path[len(img_path)-1]
        source=ET.SubElement(self.root,"source")
        ET.SubElement(source,"database").text="Moving Objects Experiment Aug.2020 PUCPR"
        ET.SubElement(source,"annotatation_user").text=annotate_user
        self.img=cv2.imread(img_file_path);
        #from IPython import embed;embed()
        height, width, depth = self.img.shape
        size=ET.SubElement(self.root,"size")
        ET.SubElement(size,"width").text=str(width)
        ET.SubElement(size,"height").text=str(height)
        ET.SubElement(size,"depth").text=str(depth)
        self.root=self.prettify(self.root)
        self.tree=ET.ElementTree(self.root)

    @staticmethod
    def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """

        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        final=ET.fromstring(reparsed.toprettyxml(indent="  "))
        return final

    def add_object(self,obj_name,coord=[]):
        obj=ET.SubElement(self.root, "object")
        ET.SubElement(obj,"name").text=str(obj_name)
        ET.SubElement(obj,"pose").text="Unspecified"
        ET.SubElement(obj,"truncated").text=str(0)
        ET.SubElement(obj,"difficult").text=str(0)
        bbox=ET.SubElement(obj,"bndbox")
        ET.SubElement(bbox,"xmin").text=str(coord[0]) #xmin
        ET.SubElement(bbox,"ymin").text=str(coord[1]) #ymin
        ET.SubElement(bbox,"xmax").text=str(coord[2]) #xmax
        ET.SubElement(bbox,"ymax").text=str(coord[3]) #ymax
        self.root=self.prettify(self.root)
        self.tree=ET.ElementTree(self.root)
        self.tree.write(self.xml_file_path)

class XMLreader(XMLer):
    def __init__(self,xml_annot_fld):
        self.xaf=xml_annot_fld
    
    def rename(self,rename_dict={}):
        for file in os.listdir(self.xaf):
            tree = ET.parse(self.xaf+"/"+file)
            root = tree.getroot()
            for obj in root.iter("object"):
                new_value=rename_dict[obj.find("name").text]
                obj.find("name").text=new_value
            
            root=self.prettify(root)
            tree=ET.ElementTree(root)
            tree.write(self.xaf+"/"+file)
    
    def add_difficult(self):
        for file in os.listdir(self.xaf):
            tree = ET.parse(self.xaf+"/"+file)
            root = tree.getroot()
            for obj in root.iter("object"):
                ET.SubElement(obj,"pose").text="Unspecified"
                ET.SubElement(obj,"truncated").text=str(0)
                ET.SubElement(obj,"difficult").text=str(0)
            
            root=self.prettify(root)
            tree=ET.ElementTree(root)
            tree.write(self.xaf+"/"+file)

class Annotate():

    def __init__(self,video_fld,image_output_fld,xml_annot_fld,
                 user_name="Luhm", video_type="out",
                 frame_skip=4,start_frame=200, max_frames=1000):
        
        self.frame_skip=frame_skip; self.start_frame=start_frame; self.max_frames=max_frames
        if video_type=="out":
            reader = Out_reader(video_fld)
            self.video_array = reader.out_2_array_complete()
        elif video_type=="bin":
            video_buffer=Bin_2_Array(video_fld).reshaped_grouped()
            for num, fs in enumerate(range(self.start_frame,self.max_frames,self.frame_skip)):

                video_slice=np.array([video_buffer[fs,:,:,:]])
                if num==0:
                    self.video_array=video_slice
                else:
                    self.video_array=np.vstack((self.video_array,video_slice))
        # video_array  [frames,fmap,240,320] 
        #from IPython import embed; embed()  
        self.video_in=video_fld
        self.iof=image_output_fld
        self.xaf=xml_annot_fld
        self.user=user_name
        ####################

    def one_frame(self, frame_n=1,    file_type="Amplitude", 
                        objects=1,  file_name="Image_",object_names=["object_1","object_2","object_3"]):
        # 0 is "Amplitude"; 1 is "Ambient"; 3 is "Depth"; 4 is "Phase"
        types={"Amplitude":0, "Ambient":1, "Depth":2, "Phase":3}
        
        frame=self.video_array[frame_n,types[file_type],:,:].copy()
        if file_type=="Amplitude":
            mask=frame>400
            frame[mask]=450
        frame=normalize_img(frame)
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
                print("Frame: ",frame_n, " Croppped Object Number: ", crp_num, " Label: ", crop_label, " coord: (",roi_coord,")")

    def all_frames(self, file_type="Amplitude", 
                        objects=1,  file_name="Image_"):

        for frame_n in range(self.video_array.shape[0]):
            #print(frame_n,self.video_array.shape)
            self.one_frame(frame_n=frame_n,    file_type=file_type, 
                            objects=objects,  file_name=file_name)

class Splitter():

    def __init__(self,path):
        ### Image and Annotation Folder
        self.annotation_p= path+"/Annotation"
        self.images_p=path+"/JPEGImages"

        ### Folder with Txt split and shuffled data
        self.dataset_p = path+"/dataset_split_layout/"
        if not os.path.exists(self.dataset_p):
            os.mkdir(self.dataset_p)

        ### Txt file names within split folder
        self.train_file_name = self.dataset_p + "train.txt"
        self.test_file_name = self.dataset_p + "test.txt"
        self.all_file_name = self.dataset_p + "train_and_test.txt"

        ### xml file list within the annotation folder
        self.files = os.listdir(self.annotation_p)

        #lists to save the names that will be save in the TXT file
        self.all_list = []; self.train_list = []; self.test_list = []

        #lists to save the image filepath test/train_images.json
        self.train_list_fullpath=[]; self.test_list_fullpath=[]
        
    def check_xml2img(self,name):
        img_path=self.images_p+"/"+name+".jpg"
        if not os.path.exists(img_path):
            img_path=self.images_p+"/"+name+".jpeg"
            if not os.path.exists(img_path):
                print("Image Annotation exist but the corresponding Image jpg pr JPEG does not")
                print("  Please check if both Annotation and Image file has the same name and ends with JPG or JPEG ")
                img_path=( "ERROR - Check the image path for the following file: "+ 
                            self.annotation_p +"/"+name)
        return img_path

    def write(self):
        # get xml files
        random.shuffle(self.files)
        random.shuffle(self.files)
        tt_files=len(self.files)
        train_num=int(len(self.files)*0.7) 
        test_num=tt_files-train_num
        for num, file in enumerate(self.files):
            if file.endswith(".xml"):
                name=file.split(".xml")[0]
                if num<train_num:
                    self.train_list.append(name)
                    self.train_list_fullpath.append(self.check_xml2img(name)) #####
                else:
                    self.test_list.append(name)
                    self.test_list_fullpath.append(self.check_xml2img(name))
                self.all_list.append(name)

        with open(self.train_file_name,"a") as filehandle:
            filehandle.writelines("%s\n" % train_name for train_name in self.train_list)
        with open(self.test_file_name,"a") as filehandle2:
            filehandle2.writelines("%s\n" % test_name for test_name in self.test_list)
        with open(self.all_file_name,"a") as filehandle3:
            filehandle3.writelines("%s\n" % all_name for all_name in self.all_list)  
        print("Saved txt files in: ", self.dataset_p)

    def read(self):
        self.all_list = []; self.train_list = []; self.test_list = []
        with open(self.train_file_name, 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                train = line[:-1] # remove linebreak which is the last character of the string
                self.train_list.append(train) # add item to the list
                self.all_list.append(train)
        with open(self.test_file_name, 'r') as filehandle:
            filecontents = filehandle.readlines()
            for line in filecontents:
                # remove linebreak which is the last character of the string
                test = line[:-1]
                # add item to the list
                self.test_list.append(test)
                self.all_list.append(test)
        print("Read txt files from: ", self.dataset_p)

class Create_Dataset():
    ### Input: Folders with images and fully annotated xml files
    #### Folder Structure: 
        ### XML files must be save here
        #""".../dataset1/Annotation/""" 
        ### Image JPG - JPEG files must be save here
        #""".../dataset1/JPEGImages/"""
        ### Txt Files dividing Train and Test shuffled images.
        #### !!! this division is only for this dataset and has to be merged with other datasets later !!!
        #""".../dataset1/dataset_split_layout/"""
        ### "bin" or "out" files with the ToF data.
        #""".../dataset1/"""
    ### Output: Folders with JSON files containing:
        ### label_map.json - dictionary with {"name": class code} ex: {"object A": 1, "background" : 0}
        ### test_images.json and train_images.json - list with full filepath for images [folderpath+img00.jpg]
        ### test_objects.json and train_objects.json - list with dictionaries
        #### !!! Each dictionary in the list is a packed information an image containing:
        ####    - bounding boxes and its classes
        #### Ex:  {"boxes": [[coordA],[coordB],....], "labels":[class codeA,class codeB]} !!!
    def __init__(self,dataset_paths, json_output,
                     labels=('cubo_gran', 'cubo_peq', 'piramide_gran', 'piramide_peq', 'esfera')):
        self.json_output=json_output
         # Label map
        self.update_labels(labels)
        ###
        self.dataset_paths=dataset_paths
        ###
        self.annotation_paths=[]

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        boxes = list()
        labels = list()
        difficulties = list()

        for object in root.iter('object'):

            difficult = int(object.find('difficult').text == '1')

            label = object.find('name').text.lower().strip()
            if label not in self.label_map:
                continue
            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text) - 1
            ymin = int(bbox.find('ymin').text) - 1
            xmax = int(bbox.find('xmax').text) - 1
            ymax = int(bbox.find('ymax').text) - 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.label_map[label])
            difficulties.append(difficult)

        return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

    def update_labels(self,labels):
        label_map = {k: v + 1 for v, k in enumerate(labels)}
        label_map['background'] = 0
        self.label_map=label_map
        self.rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
        # Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
        distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
        self.label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
        with open(self.json_output+'/label_map.json', 'w') as j:
            json.dump(self.label_map, j)  # save label map

    def split_shuffle(self):
        self.train_files=[]
        self.test_files=[] 
        for path in self.dataset_paths:
            spl=Splitter(path)
            spl.write()
            self.train_files=self.train_files+spl.train_list_fullpath
            self.test_files=self.train_files+spl.test_list_fullpath
        random.shuffle(self.train_files)
        random.shuffle(self.test_files)
                # Save Merge to JSON file
        with open(self.json_output+'/TRAIN_images.json', 'w') as j:
            json.dump(self.train_files, j)
        with open(self.json_output+'/TEST_images.json', 'w') as j:
            json.dump(self.test_files, j)

    def merge (self):
        self.split_shuffle()

        ### TRAIN ###
        n_objects = 0
        train_objects = list()

        for image_file_path in self.train_files:
            folder,f=image_file_path.split("JPEGImages")
            file=f.split("/")[1]; xmlfile=file.split(".")[0]+".xml"
            xml_filepath=folder+"/Annotation/"+xmlfile

            ###check
            objects=self.parse_annotation(xml_filepath)
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            train_objects.append(objects)
        assert len(train_objects) == len(self.train_files)
        with open(self.json_output+'/TRAIN_objects.json', 'w') as j:
            json.dump(train_objects, j)
        
        ### TEST ###
        n_objects = 0
        test_objects = list()

        for image_file_path in self.test_files:
            folder,f=image_file_path.split("JPEGImages")
            file=f.split("/")[1]; xmlfile=file.split(".")[0]+".xml"
            xml_filepath=folder+"/Annotation/"+xmlfile

            ###check
            objects=self.parse_annotation(xml_filepath)
            if len(objects['boxes']) == 0:
                continue
            n_objects += len(objects)
            test_objects.append(objects)
        assert len(test_objects) == len(self.test_files)

        with open(self.json_output+'/TEST_objects.json', 'w') as j:
            json.dump(test_objects, j)
            
if __name__ == '__main__':

    # folders = ["D:/media/ssd/ssd_data/Experimentos/bin_exp1",
    #                 "D:/media/ssd/ssd_data/Experimentos/bin_exp2",
    #                 "D:/media/ssd/ssd_data/Experimentos/bin_exp3"]

    # print(100*"#")
    # print(5*""+"Running Labeller")
    # print(100*"#")
    # ### Label / Annotate ##


    # objects_per_scene=[2,2,1]
    # renamed_objects = [{"obj_label-0":"cubo_gran","obj_label-1":"piramide_peq"},
    #                    {"obj_label-0":"cubo_peq","obj_label-1":"piramide_gran"},
    #                    {"obj_label-0":"esfera"}]

    # for num, folder in enumerate(folders):
    #     print(5*""+"Labelling folder: ", folder)
    #     print(100*"#")
    #     an=Annotate(input_fld = folder,
    #                 image_output_fld = folder+"/JPEGImages",
    #                 xml_annot_fld = folder+"/Annotation",
    #                 user_name="Luhm",video_type="bin")
    #     an.all_frames(objects=objects_per_scene[num])

    ### rename Labels / Annotate ##
    # for num, folder in enumerate(folders):
    #     print(5*""+"Renaming Standard Labels folder", folder)
    #     print(100*"#")
    #     xml = XMLreader(folder+"/Annotation")
    #     xml.rename(renamed_objects[num])

    # json_output= "D:/media/ssd/ssd_data/Experimentos/outputs"
    # # ### Create Merged Dataset with all experiments/folders

    # print(5*""+"Creating Merge Dataset Outputs")
    # print(100*"#")
    # cdts=Create_Dataset(dataset_paths = folders, json_output= json_output,
    #                  labels=('cubo_gran', 'cubo_peq', 'piramide_gran', 'piramide_peq', 'esfera'))
    # cdts.split_shuffle()
    # cdts.merge()

    from IPython import embed; embed()