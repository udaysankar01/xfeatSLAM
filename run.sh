### TESTS FOR XFEAT ###


## XFeat Extractor testing ##
# ./examples/test/test_extractor /home/uday/projects/xfeatSLAM/tgt.png


## Stereo feature in extractor testing ##
# ./examples/test/test_stereo_xf /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png \
#                                /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.211214.png

# ./examples/test/test_stereo_orb /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png \
                                # /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.211214.png


## Matching test ##
# ./examples/test/test_matcher_xf /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png \
#                                 /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.211214.png
# ./examples/test/test_matcher_xf /home/uday/projects/xfeatSLAM/ref.png \
#                                 /home/uday/projects/xfeatSLAM/tgt.png \
#                                 0.4

# ./examples/test/test_matcher_orb /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png \
#                                  /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.211214.png
# ./examples/test/test_matcher_orb /home/uday/projects/xfeatSLAM/ref.png \
#                                  /home/uday/projects/xfeatSLAM/tgt.png

#######################################################################################################################

### RGB-D WITH XFEAT AND ORB ###


## Xfeat SLAM RGB-D test ##

TH_LOW=50 TH_HIGH=100 ./examples/RGB-D/rgbd_tum /home/uday/projects/xfeatSLAM/Vocabulary/ORBvoc.txt \
                          /home/uday/projects/xfeatSLAM/examples/RGB-D/TUM3.yaml \
                          /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz \
                          /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/association.txt


## ORB SLAM RGB-D test ##

# USE_ORB=1 ./examples/RGB-D/rgbd_tum /home/uday/projects/xfeatSLAM/Vocabulary/ORBvoc.txt \
#                           /home/uday/projects/xfeatSLAM/examples/RGB-D/TUM3.yaml \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/association.txt


# ## ORB SLAM Stereo test ##
# ./examples/Stereo-Inertial/stereo_inertial_tum_vi /home/uday/projects/xfeatSLAM/Vocabulary/ORBvoc.txt \
#                                                   /home/uday/projects/xfeatSLAM/examples/Stereo-Inertial/TUM-VI.yaml \
#                                                   /home/uday/projects/xfeatSLAM/dataset-corridor4_512_16/mav0/cam0/data \
#                                                   /home/uday/projects/xfeatSLAM/dataset-corridor4_512_16/mav0/cam1/data \
#                                                   /home/uday/projects/xfeatSLAM/dataset-corridor4_512_16/mav0/cam0/data.csv \
#                                                   /home/uday/projects/xfeatSLAM/dataset-corridor4_512_16/mav0/imu0/data.csv