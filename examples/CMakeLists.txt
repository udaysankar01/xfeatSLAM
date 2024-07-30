# add_executable(test_xfeat test_xfeat.cc)
# target_link_libraries(test_xfeat
# xfeat
# ${THIRD_PARTY_LIBS}
# )
# set_property(TARGET test_xfeat PROPERTY CXX_STANDARD 17)

add_executable(test_extractor test_extractor.cc)
target_link_libraries(test_extractor
    xfeat
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET test_extractor PROPERTY CXX_STANDARD 17)