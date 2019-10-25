from fastai.vision import *
import time


# path = r'D:\fastAI\cnnModel'
# classes = ['false', 'true']
# data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.1,
#                                   size=256, bs=32, resize_method=ResizeMethod.PAD, padding_mode='zeros',
#                                   num_workers=0)
# data.normalize(imagenet_stats)
# learn = cnn_learner(data, models.resnet101, metrics=error_rate)

def gen_learn():
    path = 'images/'
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0,
                                      size=28, bs=64, num_workers=0).normalize(imagenet_stats)
    learn=cnn_learner(data,models.resnet18,metrics=error_rate)
    learn=to_fp16(learn)

    learn.load('bestmodel')
    return learn
learn = gen_learn()

def number_predict():
    start = time.time()
    img_3 = r'D:\1\rst\7-47.jpg'
    img = open_image(img_3)
    pred_class, pred_idx, outputs = learn.predict(img)
    end = time.time()
    # print(end-start)
    print(str(pred_class))


if __name__=="__main__":

    start = time.time()
    img_3 = r'D:\1\rst\7-47.jpg'
    img = open_image(img_3)
    pred_class,pred_idx,outputs = learn.predict(img)
    end = time.time()
    # print(end-start)
    print(str(pred_class))
    # print(+(-1))