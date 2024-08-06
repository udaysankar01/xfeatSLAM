add_executable(test_extractor test_extractor.cc)
target_link_libraries(test_extractor
    ${PROJECT_NAME}
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET test_extractor PROPERTY CXX_STANDARD 17)

add_executable(test_stereo_xf test_stereo_xf.cc)
target_link_libraries(test_stereo_xf
    ${PROJECT_NAME}
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET test_stereo_xf PROPERTY CXX_STANDARD 17)

add_executable(test_stereo_orb test_stereo_orb.cc)
target_link_libraries(test_stereo_orb
    ${PROJECT_NAME}
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET test_stereo_orb PROPERTY CXX_STANDARD 17)