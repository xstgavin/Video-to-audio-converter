import cv2
import json
import os
import numpy as np

def check_melgram(imgMel,imgMel16, fake_video):
    if imgMel is None or imgMel16 is None:
        return "None"
    fake_mel_path =fake_video[:-3]+'png'
    if not os.path.exists(fake_mel_path):
        print(fake_mel_path+" not exits")
    fake_mel = cv2.imread(fake_mel_path)
    diff1=  np.abs(fake_mel.astype(float) - imgMel16.astype(float))
    diff2 = np.abs(fake_mel.astype(float) - imgMel.astype(float))
    sum1 = np.sum(diff1)
    sum2 = np.sum(diff2)
    label = "FAKE"
    if sum2 <9000000:
        label="REAL"
    else:
        if sum1<9000000:
            label="REAL"
    x = np.abs(np.concatenate([diff1,diff2],axis=1)).astype(np.uint8)
    cv2.imwrite('./trial/'+label+"_"+fake_video.split('/')[-1][:-4]+"_%.2f_%.2f"%(sum1,sum2)+'.png',x)
    return label
    

jdat = json.load(open('./labels/all_label.json','r'))
pt = 'dfdc_train_part_'
fid  = open('./labels/pairs.txt','w')

root = '/home/xiaoshengtao/hdd/DATA/deepfake-detection-challenge/train_videos/'
for key in jdat.keys():
    imgMel=None
    for keyx in jdat[key].keys():
        val = jdat[key][keyx]
        partid = int(val.split('_')[-1])
        if partid < 45:
            break
        else:
            org = pt+"%d"%partid+"/"+key
            if imgMel is  None or imgMel16 is None:
                melName = root+org[:-3]+'png'
                mel16 = root+pt+"%d/mel/"%partid+key[:-4]+"_16000.png"
                imgMel = cv2.imread(melName)
                imgMel16 = cv2.imread(mel16)

            fake = root+pt+"%d"%partid+"/"+keyx
            label = check_melgram(imgMel,imgMel16,fake)
            fid.write(org+' '+fake+' '+label+'\n')
fid.close()
