add_executable(test test.cc)

target_link_libraries(test
    xfeat
    ${THIRD_PARTY_LIBS}
)

set_property(TARGET test PROPERTY CXX_STANDARD 17)