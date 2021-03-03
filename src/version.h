//
// Created by jreuter on 01.03.21.
//

#ifndef PLIMIG_VERSION_H_IN
#define PLIMIG_VERSION_H_IN

#include <string>
#include <sstream>
#include <ctime>
#include <chrono>

namespace PLImg {
    class Version {

    public:
        /**
         * Returns the last commit hash of the git project. If no git project is present return N/A instead.
         * The commit hash will end with + if there are changes beyond the current commit which weren't committed yet.
         * @return String with the git hash of the last commit
         */
        inline static std::string versionHash() {
            return Version::PROJECT_VER_HASH;
        }

        /**
         * Returns the complete project version defined through CMake
         * @return String with the current project version
         */
        inline static std::string projectVersion() {
            return Version::PROJECT_VER;
        }

        /**
         * Returns the current project major version defined through CMake
         * @return String with the current major version
         */
        inline static std::string projectVersionMajor() {
            return Version::PROJECT_VER_MAJOR;
        }

        /**
         * Returns the current project minor version defined through CMake
         * @return String with the current minor version
         */
        inline static std::string projectVersionMinor() {
            return Version::PROJECT_VER_MINOR;
        }

        /**
         * Returns the current project patch version defined through CMake
         * @return String with the current patch version
         */
        inline static std::string projectVersionPatch() {
            return Version::PROJECT_VER_PATCH;
        }

        /**
         * Returns the current project name defined through CMake
         * @return String with the current project name
         */
        inline static std::string projectName() {
            return Version::PROJECT_NAME;
        }

        /**
         * Returns the current project branch defined through Git. If no git project is present, the return value will
         * be N/A instead.
         * @return String with the current git branch
         */
        inline static std::string projectBranch() {
            return Version::PROJECT_VER_BRANCH;
        }

        /**
         * Returns the current project tag defined through Git
         * @return Current project tag. Empty if no project tag is given. N/A if no git project is present.
         */
        inline static std::string projectTag() {
            return Version::PROJECT_VER_TAG;
        }

        /**
         * Generates current time stamp according to the current time on the executing machine
         * @return String with time and date.
         */
        static std::string timeStamp() {
            time_t time = std::chrono::system_clock::to_time_t(std::chrono::high_resolution_clock::now());
            return std::ctime(&time);
        }

    private:
        /*
         * Note: Those variables will be set through the version.cpp.in file. When calling CMake, the version.cmake
         * file in ${PROJECT_SOURCE_DIR}/cmake will be called. Here all relevant variables will be extracted. The
         * main CMakeLists.txt will transfer those variables to the version.cpp.in file and write a new version.cpp file
         * which is used for compilation.
         */
        static const std::string PROJECT_NAME;
        static const std::string PROJECT_VER;
        static const std::string PROJECT_VER_MAJOR;
        static const std::string PROJECT_VER_MINOR;
        static const std::string PROJECT_VER_PATCH;
        static const std::string PROJECT_VER_HASH;
        static const std::string PROJECT_VER_BRANCH;
        static const std::string PROJECT_VER_TAG;
    };
}

#endif