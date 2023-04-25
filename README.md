<div align="center">
   <h1>Parking System using YOLOv5 and SORT</h1>
<p>
<!--    <a align="left" href="https://ultralytics.com/yolov5" target="_blank"> -->
   <img width="850" src="https://github.com/janleven01/parkingSystem_YOLOV5_SORT/blob/ad5e0fe4fc04a8207e32a14bc637c17bad154c62/ParkingSystem.jpg"></a>
</p>
<br>
<div>
<!--    <a href="https://youtu.be/fIcd6eE700M"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="Promotional Video"></a> -->
<!--    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
   <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
   <br>
   <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
   <a href="https://join.slack.com/t/ultralytics/shared_invite/zt-w29ei8bp-jczz7QYUmDtgo6r6KcMIAg"><img src="https://img.shields.io/badge/Slack-Join_Forum-blue.svg?logo=slack" alt="Join Forum"></a> -->
</div>
<br>
<div align="center">
<!--    <a href="https://github.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.linkedin.com/company/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://twitter.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="2%"/>
   </a> -->
   <img width="5%" />
   <a href="https://youtu.be/fIcd6eE700M">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="5%"/>
   </a>
<!--    <img width="2%" />
   <a href="https://www.facebook.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.instagram.com/ultralytics/">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="2%"/>
   </a> -->
</div>

<br>
<p>
This project develops a smart parking system with a deep learning approach to detect available/unavailable parking spaces using surveillance cameras for cheaper but detailed parking information; this approach also includes vehicle type detection, tracking, and color detection.
</p>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Abstract</div>
<p align="justify">
Convenience is one of the best things about owning a car, but that convenience outweighs the frustration of roaming around looking for available parking spaces in a busy parking lot. It also costs valuable resources like time, money, and effort. This study developed a smart parking system with a deep learning approach to detect available/unavailable parking spaces using surveillance cameras for cheaper but detailed parking information. The pre-trained Yolov5 models used were YoloV5m and YoloV5l. All results of the model evaluation have a high level of precision and recall. The Yolov5l model has higher precision of 98.6% compared to Yolov5m which has 97.2%. Despite the performance, the trained models incorrectly classify the car type occasionally. The three types - SUV, Sedan, and Minivan, cause the incorrect classifications to each other due to its similarity at certain angles as the IP camera is in an elevated position. Furthermore, color detection results vary over time because of the existing shadows and the intensity of light causing a reflection, as cars have reflective surfaces, confusing the feature extractors to read it as white. Nonetheless, the occupancy identification of parking spaces has state-of-the-art functionality with the accuracy of 100%. This study recommended improving color detection which can accurately detect colors under various lighting conditions to improve detection’s robustness.
</p>


</div>
