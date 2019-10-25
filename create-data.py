import os
import cv2
import cv2 as cv
import numpy as np
import math
import random
import json
from xml.dom.minidom import parseString
# import tensorflow as tf
import xmldict

def get_json_dict(list_class, list_b, imagePath, savepath):
    f = open("template.json", encoding='utf-8')
    shapes = []
    for i,v in enumerate(list_b):

         shape = {}
         shape['label'] = str(list_class[i])
         shape['line_color'] = None
         shape['fill_color'] = None
         shape['points'] = for_list(v)
         shapes.append(shape)
    jsonData = json.load(f)
    jsonData['shapes'] = shapes
    jsonData['imagePath'] = imagePath
    with open(savepath+'\\'+imagePath.replace('jpg', 'json'), 'w') as w:
        json.dump(jsonData, w)
    f.close()

def get_json_dict1(list_class, list_b, imagePath, savepath):
    f = open("template.json", encoding='utf-8')
    shapes = []
    for i,v in enumerate(list_b):

         shape = {}
         # shape['label'] = str(list_class[i])
         shape['label'] = 'number'
         shape['line_color'] = None
         shape['fill_color'] = None
         shape['points'] = v
         shapes.append(shape)
    jsonData = json.load(f)
    jsonData['shapes'] = shapes
    jsonData['imagePath'] = imagePath
    with open(savepath+'\\'+imagePath.replace('jpg', 'json'), 'w') as w:
        json.dump(jsonData, w)
    f.close()


def get_xml_write(list_class, list_b, imagePath, savepath):
    with open('template.xml', 'r') as fid:
        xml_str = fid.read()
    # xml_dict = xmldict.xml_to_dict(xml_str)['annotation']['object']
    xml_dict = xmldict.xml_to_dict(xml_str)
    del xml_dict['annotation']['object']
    objects = []
    for i, v in enumerate(list_b):
        object = {}
        object['name'] = list_class[i]

        object['truncated'] = 0
        object['difficult'] = 0
        object['pose'] = 'Unspecified'
        bndbox = {'xmin': str(v[0]), 'ymin': str(v[1]), 'xmax': str(v[2]), 'ymax': str(v[3])}
        object['bndbox'] = bndbox
        objects.append(object)

    xml_dict['annotation']['object'] = objects
    dict_xml = xmldict.dict_to_xml(xml_dict)

    dom = parseString(dict_xml).toprettyxml()
    # print(xml_dict['annotation']['folder'])

    with open(savepath+'\\'+imagePath.replace('jpg', 'xml'), 'w') as f:
        f.write(dom)
        f.close()

    # print(xmldict)

def get_xml_write1(classname, box, imagePath, savepath):
    with open('template.xml', 'r') as fid:
        xml_str = fid.read()
    # xml_dict = xmldict.xml_to_dict(xml_str)['annotation']['object']
    xml_dict = xmldict.xml_to_dict(xml_str)
    del xml_dict['annotation']['object']
    object = {}
    object['name'] = classname
    object['truncated'] = 0
    object['difficult'] = 0
    object['pose'] = 'Unspecified'
    bndbox = {'xmin': str(box[0]), 'ymin': str(box[1]), 'xmax': str(box[0]+box[2]), 'ymax': str(box[1]+box[3])}
    object['bndbox'] = bndbox
    # [x, y, x + w, y + h]
    xml_dict['annotation']['object'] = object
    dict_xml = xmldict.dict_to_xml(xml_dict)

    dom = parseString(dict_xml).toprettyxml()

    with open(savepath+'\\'+imagePath.replace('jpg', 'xml'), 'w') as f:
        f.write(dom)
        f.close()

def for_list(arrlist):
    super_list = []
    for i in arrlist.tolist():
        super_list.append(i[0])
    return super_list

def for_list1(arrlist,k=0):
    super_list = []
    for i in arrlist:
        if len(i.tolist()) < 5:
            return
        for j in i.tolist():
            j[0][0]+=k
            super_list.append(j[0])
        # super_list[len(super_list)-1] = i.tolist()[0][0]
    return super_list

def get_data(filepath):
    data_path = os.listdir(filepath)
    data_dict = {}
    for path in data_path:
        data_list = os.listdir(filepath+"\\"+path)
        list_data = []
        for data in data_list:
             list_data.append(filepath+"\\"+path+"\\"+data)
        data_dict[path] = list_data
    return data_dict

def create_data(time, num, filepath,savepath,create_class):
    data_dict = get_data(filepath)
    for n in range(num):
        for i in range(time):
            img = np.ones((92, 170), dtype=np.uint8)
            if i < 10:
                list_data = data_dict[str(i)]
                img1 = cv2.imread(list_data[random.randint(0,len(list_data)-1)],0)
                rows, cols = img1.shape[:2]
                roi = img[:rows, :cols]

                # 创建掩膜
                #img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                # 保留除logo外的背景
                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                dst = cv2.add(img1_bg, img1)  # 进行融合
                img[:rows, :cols] = dst
                img_mask = cv2.bitwise_not(img,img)
                #生成xml
                # list_class = [str(i)]
                # list_b = [get_outrect(img1)]
                rect = getBox(img_mask)
                get_xml_write1(str(i), rect, '%i-%i.jpg' % (i, n), savepath)
                cv2.imwrite(savepath+'\\%i-%i.jpg'%(i,n), img_mask)
            elif i<100 and i>9:
                list_class = [str(i)[:1],str(i)[-1]]
                i1 = data_dict[str(i)[:1]]
                i2 = data_dict[str(i)[-1]]
                list_arr = [i1,i2]
                k = 0
                for g,j in enumerate(list_arr):

                    img1 = cv2.imread(j[random.randint(0, len(j) - 1)], 0)
                    rows, cols = img1.shape[:2]
                    roi = img[:rows, k:cols+k]

                    # 创建掩膜
                    # img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # 保留除logo外的背景
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    dst = cv2.add(img1_bg, img1)  # 进行融合
                    img[:rows, k:cols+k] = dst
                    k += 15
                img_mask = cv2.bitwise_not(img, img)
                rect = getBox(img_mask)
                get_xml_write1(str(i), rect, '%i-%i.jpg' % (i, n), savepath)
                cv2.imwrite(savepath + '\\%i-%i.jpg' % (i, n), img_mask)
            else:
                i1 = data_dict[str(i)[:1]]
                i2 = data_dict[str(i)[1:2]]
                i3 = data_dict[str(i)[-1]]

                list_arr = [i1, i2, i3]
                k = 0
                for j in list_arr:

                    img1 = cv2.imread(j[random.randint(0, len(j) - 1)], 0)
                    rows, cols = img1.shape[:2]
                    roi = img[:rows, k:cols+k]

                    # 创建掩膜
                    # img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # 保留除logo外的背景
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    dst = cv2.add(img1_bg, img1)  # 进行融合
                    img[:rows, k:cols+k] = dst
                    k += 15
                img_mask = cv2.bitwise_not(img, img)
                rect = getBox(img_mask)
                get_xml_write1(str(i), rect, '%i-%i.jpg' % (i, n), savepath)
                cv2.imwrite(savepath + '\\%i-%i.jpg' %(i,n), img_mask)

def getBox(img):
    # img=cv2.imread(path,0)
    _,th=cv2.threshold(img,128,255,cv2.THRESH_BINARY_INV)
    kernal=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernal,iterations=3)

    _,contours,hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cnts=contours[0]
    if len(contours)>1:
        for i in range(1,len(contours)):
            cnts=np.vstack((cnts, contours[i]))
    rect=cv2.boundingRect(cnts)
    # p1,p2=(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3])
    # cv2.rectangle(img,p1,p2,(0,0,0))
    return rect

def create_data1(time, num, filepath,savepath,create_class):
    data_dict = get_data(filepath)
    for n in range(num):
        for i in range(time):
            img = np.ones((92, 170), dtype=np.uint8)
            if i < 10:
                list_data = data_dict[str(i)]
                img1 = cv2.imread(list_data[random.randint(0,len(list_data)-1)],0)
                if create_class == 'json':
                    list_class = [str(i)]
                    con = get_outline(img1)
                    list_arr = for_list1(con,0)
                    if list_arr == None :
                        continue
                    list_b = [list_arr]
                    get_json_dict1(list_class, list_b,  '%i-%i.jpg'%(i,n), savepath)
                else:
                    list_class = [str(i)]
                    list_b = [get_outrect(img1)]
                    get_xml_write(list_class, list_b,  '%i-%i.jpg'%(i,n), savepath)
                rows, cols = img1.shape[:2]
                roi = img[:rows, :cols]

                # 创建掩膜
                #img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)

                # 保留除logo外的背景
                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                dst = cv2.add(img1_bg, img1)  # 进行融合
                img[:rows, :cols] = dst
                img_mask = cv2.bitwise_not(img,img)
                cv2.imwrite(savepath+'\\%i-%i.jpg'%(i,n), img_mask)
            elif i<100 and i>9:
                list_class = [str(i)[:1],str(i)[-1]]
                i1 = data_dict[str(i)[:1]]
                i2 = data_dict[str(i)[-1]]
                list_arr = [i1,i2]
                k = 0
                list_b = []
                for g,j in enumerate(list_arr):

                    img1 = cv2.imread(j[random.randint(0, len(j) - 1)], 0)
                    if create_class == 'json':
                        con = get_outline(img1)
                        if len(con) == 0:
                            list_b == []
                            return
                        list_b .append(for_list1(con[0], 0))
                    else:
                        rect = get_outrect(img1)
                        rect[0] +=k
                        rect[2] +=k
                        list_b.append(rect)

                    rows, cols = img1.shape[:2]
                    roi = img[:rows, k:cols+k]

                    # 创建掩膜
                    # img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # 保留除logo外的背景
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    dst = cv2.add(img1_bg, img1)  # 进行融合
                    img[:rows, k:cols+k] = dst
                    k += 15
                if len(list_b) == 0:
                    return
                if create_class == 'json':
                    get_json_dict1(list_class, list_b, '%i-%i.jpg' % (i, n), savepath)
                else:
                    get_xml_write(list_class,list_b, '%i-%i.jpg' % (i, n), savepath)
                img_mask = cv2.bitwise_not(img, img)
                cv2.imwrite(savepath + '\\%i-%i.jpg' %(i,n), img_mask)
            else:
                list_class = [str(i)[:1], str(i)[1:2], str(i)[-1]]
                i1 = data_dict[str(i)[:1]]
                i2 = data_dict[str(i)[1:2]]
                i3 = data_dict[str(i)[-1]]

                list_arr = [i1, i2, i3]
                list_b = []
                k = 0
                for j in list_arr:

                    img1 = cv2.imread(j[random.randint(0, len(j) - 1)], 0)
                    if create_class == 'json':
                        con = get_outline(img1)
                        if len(con) == 0:
                            list_b == []
                            return
                        list_b.append(for_list1(con[0], 0))
                    else:
                        rect = get_outrect(img1)
                        rect[0] += k
                        rect[2] += k
                        list_b.append(rect)
                    rows, cols = img1.shape[:2]
                    roi = img[:rows, k:cols+k]

                    # 创建掩膜
                    # img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img1, 30, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    # 保留除logo外的背景
                    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    dst = cv2.add(img1_bg, img1)  # 进行融合
                    img[:rows, k:cols+k] = dst
                    k += 15
                if len(list_b) == 0:
                    return
                if create_class == 'json':
                    get_json_dict1(list_class, list_b, '%i-%i.jpg' % (i, n), savepath)
                else:
                    get_xml_write(list_class,list_b, '%i-%i.jpg' % (i, n), savepath)
                img_mask = cv2.bitwise_not(img, img)
                cv2.imwrite(savepath + '\\%i-%i.jpg' %(i,n), img_mask)

def get_outline(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    cv.imwrite('binary.jpg', binary)
    _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    # x, y, w, h = cv2.boundingRect(cnt)
    # img1 = binary[x: x + w,y:y + h]

    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # print(contours[0].tolist())
    # cv2.imshow("img", img)
    #cv2.waitKey(0)
    return contours

def get_outrect(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
    # cv.imwrite('binary.jpg', binary)
    _,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for j,i in enumerate(contours):
    #     if len(i)<7:
    #         contours.pop(j)
    #         break

    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)

    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv.imwrite('img.jpg', img)
    # print(contours[0])
    #del contours[0]
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # print(contours[0].tolist())
    # cv2.imshow("img", img)
    # print([x,y,x+w,y+h])
    #cv2.waitKey(0)
    return [x,y,x+w,y+h]

# def  get_image(img):

def get_outrect1(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    result = cv2.blur(img, (5, 5))
    ret, binary = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for j,i in enumerate(contours):
    #     if len(i)<7:
    #         contours.pop(j)
    #         break
    list_rect = []
    for contour in contours:
        x, y , w, h=cv2.boundingRect(contour)
        list_rect.append([x,y,x+w,y+h])


    # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv.imwrite('img.jpg', img)
    # print(contours[0])
    #del contours[0]
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    # print(contours[0].tolist())
    # cv2.imshow("img", img)
    # print([x,y,x+w,y+h])
    #cv2.waitKey(0)

    return list_rect

def get_xml_write2(classname, box, savepath):
    with open('template.xml', 'r') as fid:
        xml_str = fid.read()
    # xml_dict = xmldict.xml_to_dict(xml_str)['annotation']['object']
    xml_dict = xmldict.xml_to_dict(xml_str)
    del xml_dict['annotation']['object']
    objects = []
    for i in box:
        object = {}
        object['name'] = classname
        object['truncated'] = 0
        object['difficult'] = 0
        object['pose'] = 'Unspecified'
        bndbox = {'xmin': str(i[0]), 'ymin': str(i[1]), 'xmax': str(i[2]), 'ymax': str(i[3])}
        object['bndbox'] = bndbox
        objects.append(object)
    # [x, y, x + w, y + h]
    xml_dict['annotation']['object'] = objects
    dict_xml = xmldict.dict_to_xml(xml_dict)

    dom = parseString(dict_xml).toprettyxml()

    with open(savepath, 'w') as f:
        f.write(dom)
        f.close()

def create_rect_text(filename,class_name,savepath,list_rect):
    with open(savepath.replace('image', 'label') + '\\%s.txt' % filename, 'w') as f:
        for txt in list_rect:
            # f.write('%i,%i,%i,%i,%i,%i,%i,%i\n'%(text[0],text[1],text[2],text[1],text[3],text[0],text[2],text[3]))
            f.write('%i,%i,%i,%i\n' % (txt[0], txt[1], txt[2], txt[3]))

def create_text(num,number, filepath, savepath):
    data_dict = get_data(filepath)
    for i in range(num):
        for j in range(number):
            list_rect = []
            img = np.ones((95, 170), dtype=np.uint8)
            if j < 10:
                k = 0
                list_class = [str(j)]
                list_data = data_dict[str(j)]
                img1 = cv2.imread(list_data[random.randint(0,len(list_data)-1)],0)
                img1 = cv.resize(img1,(58,58), interpolation=cv.INTER_AREA)
                rows, cols = img1.shape[:2]
                rows1, cols1 = img.shape[:2]
                excursion_rows1 = math.ceil(rows1/2 - rows/2)
                excursion_cols1 = math.ceil(cols1/2 - cols/2)+ k
                imgcc=cv2.bitwise_or(img[excursion_rows1:rows+excursion_rows1, excursion_cols1:cols + excursion_cols1], img1)
                img[excursion_rows1:rows + excursion_rows1, excursion_cols1:cols + excursion_cols1] = imgcc

                #img_mask = cv2.bitwise_and(img, imgcc)

                list_rect = get_outrect1(img)
                get_xml_write2(str(j),list_rect,savepath.replace('image','label')+'\\%i-----%i.xml'% (i,j))
                # create_rect_text(str(i)+"-"+str(j),list_class,savepath,list_rect)
                ret, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
                cv.imwrite(savepath + '\\%i-----%i.jpg' % (i,j), mask)

            elif j<100 and j>9:
                list_class = [str(j)[:1], str(j)[-1]]
                k = 0
                for n in list_class:
                    list_data = data_dict[n]
                    img1 = cv2.imread(list_data[random.randint(0, len(list_data) - 1)], 0)  ##黑底白字
                    img1 = cv.resize(img1, (48, 48), interpolation=cv2.INTER_LINEAR)

                    ##第二张图  黑字
                    ## 放第一张图
                    ##大图的第二个区域     和第二张图  and
                    ##

                    rows, cols = img1.shape[:2]
                    rows1, cols1 = img.shape[:2]
                    excursion_rows1 = math.ceil(rows1 / 2 - rows / 2)
                    excursion_cols1 = math.ceil(cols1 / 2 - cols / 2)+ k
                    imgcc = cv2.bitwise_or(
                        img[excursion_rows1:rows + excursion_rows1, excursion_cols1:cols + excursion_cols1], img1)

                    img[excursion_rows1:rows + excursion_rows1, excursion_cols1:cols + excursion_cols1] = imgcc

                    #融合
                    # img_mask = cv2.bitwise_and(img, img)

                    k+=30
                list_rect = get_outrect1(img)
                create_rect_text(str(i)+"-"+str(j),list_class, savepath, list_rect)

                ret, mask = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY_INV)
                cv.imwrite(savepath + '\\%i-%i.jpg' % (i,j), mask)

            else:
                list_data = data_dict[str(j)]
                img1 = cv2.imread(list_data[random.randint(0, len(list_data) - 1)], 0)

                rows, cols = img1.shape[:2]
                rows1, cols1 = img.shape[:2]
                excursion_rows1 = int(rows1 / 2 - rows / 2)
                excursion_cols1 = int(cols1 / 2 - cols / 2) + k
                img[excursion_rows1:rows + excursion_rows1, excursion_cols1:cols + excursion_cols1] = img1
                img_mask = cv2.bitwise_and(img, img)
                list_rect = get_outrect1(img_mask)
                create_rect_text(j, savepath, list_rect)
                ret, mask = cv2.threshold(img_mask, 60, 255, cv2.THRESH_BINARY_INV)
                cv.imwrite(savepath + '\\%i.jpg' % j, mask)


def get_img(img):
    ret, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    # kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    #
    # cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernal, iterations=3)
    cv.imwrite('binary.jpg', binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for j,i in enumerate(contours):
    #     if len(i)<7:
    #         contours.pop(j)
    #         break
    for i,contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # ret, th = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        img1 = binary[ y:y + h,x:x + w]

        saveimg = cv.resize(img1, (28, 28), interpolation=cv.INTER_AREA)
        cv.imwrite(r'D:\ocr_data\1\8%i.jpg'%i,saveimg)


import shutil
def copy_img(path):
    listpath = os.listdir(path)
    for i in range(300):
        shutil.copy(path + '\\' + listpath[random.randint(0, len(listpath) - 1)],  'D:\\ocr_data\\2\\copy9' + str(i) + '.png')

def resize_img(path):
    listpath = os.listdir(path)
    for path1 in listpath:
        listnum = os.listdir(path+'\\'+path1)
        for num in listnum:
            img = cv.imread(path+'\\'+path1+'\\'+num)
            img1 = cv.resize(img, (38, 38), interpolation=cv.INTER_AREA)
            cv.imwrite(path+'\\'+path1+'\\'+num,img1)

if __name__ == '__main__':
    # path = r'D:\1'
    # listImg = os.listdir(path)
    # for imgpath in listImg:
    #     img = cv.imread(path+'\\'+imgpath)
    #     ret, mask = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    #     img1 = cv.resize(mask, (28, 28), interpolation=cv.INTER_AREA)
    #     cv2.imwrite('D:\\2\\%s'%imgpath,img1)
    # img = cv.imread(r'D:\ocr_data\4.jpg',0)
    # get_img(img)
    # resize_img(r'D:\ocr_data\2')
    copy_img(r'D:\ocr_data\1')
    #get_xml_write([1],[[2,8,7,6]])
    # img = cv.imread(r'D:\ocr_data\save_data\9-0.jpg')
    # get_outrect(img)
    #print(get_data(r'D:\ocr_data\data')['0'])
    # create_data1(10, 5000, r'D:\ocr_data\1\train', r'D:\DataSet\FOXCONN\image','json')
    # create_text(300,10, r'D:\ocr_data\train', r'D:\ocr_data\image')
    # str.replace()
    #print(str(120)[1:2])
    #img = np.ones((92, 170), dtype=np.uint8)
    # img = cv.imread(r'D:\ocr_data\save_data\9-0.jpg')
    # get_outline(img)
    #get_json_dict('1','1')\

    # img = np.ones((92, 170), dtype=np.uint8)
    # img1=cv2.imread(r'D:\ocr_data\data\0\297.png',0)
    # img2=cv2.imread(r'D:\ocr_data\data\1\2.png',0)
    #
    # h,w=img1.shape
    # img[10:10+h,10:10+w]=img1
    #
    # ret, mask = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask)
    # cv2.imwrite('mask.jpg', mask)
    # roi=img[10:10+h,30:30+w]
    #
    # roi=cv2.bitwise_or(roi,roi,mask)
    #
    # cv2.imwrite('all.jpg',img)
    # print(getBox(r'D:\ocr_data\18-0.jpg'))




