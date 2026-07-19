cmake_minimum_required (VERSION 3.20)

set(targetname queue_scale_study)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")
add_executable(${targetname})
set_property(TARGET ${targetname} PROPERTY EXCLUDE_FROM_ALL TRUE)
target_sources(${targetname} PRIVATE ${sources})
target_link_libraries(${targetname} PRIVATE DynaPlex::DynaPlex)
