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


## ORB SLAM RGB-D test ##

# USE_ORB=1 ./examples/RGB-D/rgbd_tum /home/uday/projects/xfeatSLAM/Vocabulary/ORBvoc.txt \
#                           /home/uday/projects/xfeatSLAM/examples/RGB-D/TUM3.yaml \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/association.txt

# USE_ORB=1 ./examples/RGB-D/rgbd_tum /home/uday/projects/xfeatSLAM/Vocabulary/ORBvoc.txt \
#                           /home/uday/projects/xfeatSLAM/examples/RGB-D/TUM3.yaml \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_rpy \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_rpy/association.txt

## Xfeat SLAM RGB-D test ##

TH_LOW=100 TH_HIGH=1000 ./examples/RGB-D/rgbd_tum /home/uday/projects/xfeatSLAM/Vocabulary/ORBvoc.txt \
                          /home/uday/projects/xfeatSLAM/examples/RGB-D/TUM3.yaml \
                          /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz \
                          /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_xyz/association.txt

# TH_LOW=100 TH_HIGH=1000 ./examples/RGB-D/rgbd_tum /home/uday/projects/xfeatSLAM/Vocabulary/ORBvoc.txt \
#                           /home/uday/projects/xfeatSLAM/examples/RGB-D/TUM3.yaml \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_rpy \
#                           /home/uday/projects/xfeatSLAM/rgbd_dataset_freiburg1_rpy/association.txt