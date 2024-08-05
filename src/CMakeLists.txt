add_library(xfeat
    XFeat.cc
    XFextractor.cc
    ORBextractor.cc
)

set_property(TARGET xfeat PROPERTY CXX_STANDARD 17)

target_link_libraries(
    xfeat
    ${THIRD_PARTY_LIBS}
)
