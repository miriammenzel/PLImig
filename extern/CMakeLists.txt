find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    set(default_build_type "Debug")

    # Update submodules as needed
    option(PLIMG_SUBMODULE "Check submodules during build" ON)
    if(PLIMG_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(
                COMMAND ${GIT_EXECUTABLE} submodule update --init
                WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
                RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(
                    FATAL_ERROR
                    "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout submodules"
            )
        endif()
    endif()
endif()