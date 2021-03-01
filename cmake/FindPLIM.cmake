find_path(PLIM_INCLUDE_DIRS
        AttributeHandler.h
        HINTS
        /opt/work/include/plim
        /usr/local/include/plim
        /usr/include/plim
)

find_library(PLIM_LIBRARIES
        NAMES libplim.so libplim.a plim.lib
        HINTS
        ${PLIM_INCLUDE_DIRS}/../../lib/
)

if (PLIM_INCLUDE_DIRS AND PLIM_LIBRARIES)
    set(PLIM_INCLUDE_DIRS ${PLIM_INCLUDE_DIRS}/../)
    set(PLIM_FOUND TRUE)
    IF( NOT PLIM_FIND_QUIETLY )
        MESSAGE(STATUS "Found PLIM: ${PLIM_LIBRARIES}")
    ENDIF ()

else ()
    set(PLIM_FOUND FALSE)
    IF( PLIM_FIND_REQUIRED )
        MESSAGE(FATAL_ERROR "Could not find PLIM (which is required)")
    ENDIF()
endif()
