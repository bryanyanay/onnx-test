import onnxruntime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

session = onnxruntime.InferenceSession("pspnet.onnx", None)

for i in range(7,8):
  origImg = Image.open(f"images/{i+1}.JPG")
  img = origImg.resize((1024, 2048))

  imgArr = np.array(img).transpose(2, 1, 0)
  imgArr = imgArr.reshape(1, 3, 1024, 2048).astype('float32')

  raw_result = np.array(session.run([], {"input": imgArr}))
  result = raw_result.squeeze().astype("uint8")

  resImg = Image.fromarray(result)

  fig, ax = plt.subplots()

  ax.imshow(origImg)
  ax.imshow(resImg.rotate(-90).resize(origImg.size), alpha=0.5) # 
  plt.show()  

  """ IF I USE THIS CODE, THE SAVED IMG IS JUST THE ORIGINAL IMG, OTHERWISE JUST BLACK IMAGE
    if resImg.mode != 'RGB':
    resImg = img.convert('RGB')
  """
  resImg.save(f"results/{i+1}_output.jpg")
