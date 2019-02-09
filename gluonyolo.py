# See https://gluon-cv.mxnet.io/build/examples_detection/demo_yolo.html#sphx-glr-build-examples-detection-demo-yolo-py
from gluoncv import model_zoo, data, utils
import datetime
import math
import random, string
from time import gmtime, strftime
import os
from six.moves import urllib
import time
import psutil
from time import gmtime, strftime

# YOLO / Gluon / Apache MXNet
#
def yolo(args):
  
  url = args["url"]
  
  start = time.time()
  filename = '/tmp/gluoncv_image.jpg'
  filepath2 = filename
  filepath2, _ = urllib.request.urlretrieve(url, filepath2)

  net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

  x, img = data.transforms.presets.yolo.load_test(filename, short=512)
  class_IDs, scores, bounding_boxs = net(x)

  classname = str(class_IDs[0,0,0][0]).strip('<NDArray 1 @cpu(0)>') 
  classname = classname.strip('\n[')
  classname = classname.strip('.]\n')
  classname = int(classname)

  matchingpct = str(scores[0][0][0]).strip('<NDArray 1 @cpu(0)>') 
  matchingpct = matchingpct.strip('\n[')
  matchingpct = matchingpct.strip('.]\n')
  matchingpct = float(matchingpct)

  end = time.time()
  row = { }
  row['class1'] = str(net.classes[classname])
  row['pct1'] = '{0}'.format( str((matchingpct) * 100))
  row['host'] = os.uname()[1]
  row['shape'] = str(x.shape)
  row['end'] = '{0}'.format( str(end ))
  row['te'] = '{0}'.format(str(end-start))
  row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
  row['cpu'] = psutil.cpu_percent(interval=1)
  row['memory'] = psutil.virtual_memory().percent
  result = row

  return result
