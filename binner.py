import numpy as np
import os

class Bin_2_Array():

    def __init__(self,input_fld):
        """Initial parameters for Texas Tof Model ['OPT8241']"""
        self.input_folder=input_fld
        self._rename_amplitude()
        print("Reading files from ... "+input_fld)
        #### Adjust according to Kit Resolution - Standard is Texas Kit = 240x320 pixels
        self.device_resol=[240,320] 
        self.tt_points_per_frame=self.device_resol[0]*self.device_resol[1]
        self.path_list=[]
        self.file_dict={}
        self._file_type_dict()
        self.file_list(show_file=False)

    def _file_type_dict(self):
        file_type_List=['Ambient',
        '0Amplitude','AmplitudeAvg','AmplitudeStd',
        'Depth','DepthAvg','DepthStd','Distance',
        'Phase','PhaseAvg','PhaseStd',
        'PointCloud']

        dtype_list=[np.uint8,
        np.uint16,np.uint16,np.uint16,
        np.float32,np.float32,np.float32,np.float32,
        np.uint16,np.uint16,np.uint16,
        np.float32]
        
        for i in range(len(file_type_List)):
            filedetails={}
            filetype=file_type_List[i]
            dtype=dtype_list[i]
            filedetails['dtype']=dtype
            if filetype!='PointCloud' and filetype in file_type_List:
                filedetails['map']=1
            elif filetype=='PointCloud':
                filedetails['map']=4
            self.file_dict[filetype]=filedetails

    def _rename_amplitude(self):
        valid_files = ".bin"
        for file in os.listdir(self.input_folder):
            if (file.endswith(valid_files)and file.split('.')[0]=="Amplitude"):
                old=self.input_folder+'/'+file
                new=self.input_folder+'/0'+file
                os.rename(old, new)
                
    def file_list(self,show_file=False):
        valid_files = ".bin"
        for file in os.listdir(self.input_folder):
            if file.endswith(valid_files):
                n_path=os.path.join(self.input_folder,file)
                if n_path not in self.path_list:
                    self.path_list.append(n_path)
        if show_file==True:
            print("files in the selected folder: ")
            print(self.input_folder)

    def _raw_single_file(self,file):
        _,b=os.path.split(file)
        filetype,_=os.path.splitext(b)
        #print("file name =",file," filetype =",filetype)
        if filetype in self.file_dict:
            dtype=self.file_dict[filetype]['dtype']
            fmap=self.file_dict[filetype]['map']
            data=np.fromfile(file,dtype=dtype)
        return data,filetype,fmap

    def _reshape_data(self,data,filetype,fmap):
        # Reshaping data
        if fmap==1:
            frames=int(len(data)/self.tt_points_per_frame)
            data=data.reshape(frames,self.device_resol[1],self.device_resol[0],fmap).swapaxes(-1,-3)
        elif fmap==4:
            frames=int(len(data)/(self.tt_points_per_frame*fmap))
            data=data.reshape(frames,fmap,self.device_resol[0],self.device_resol[1])
        return data

    def reshaped_single(self,file):
        data,filetype,fmap=self._raw_single_file(file)
        reshaped_data=self._reshape_data(data,filetype,fmap)
        return reshaped_data,filetype,fmap

    def reshaped_grouped(self):
        count=0
        self.file_list(self.input_folder) #### solve similiar problem
        for file in self.path_list:
            data,filetype,fmap=self._raw_single_file(file)
            reshaped_data=self._reshape_data(data,filetype,fmap)
            if count==0:
                #print("count start",count)
                dataGroup=reshaped_data
                #print("data shape",dataGroup.shape)
            else:
                dataGroup=np.hstack((dataGroup,reshaped_data))
            count+=1
            #print("data shape",dataGroup.shape,"count",count,"- file_type=",filetype)

        return dataGroup