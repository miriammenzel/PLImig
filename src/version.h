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
        inline static std::string string() {
            // Variable is set by cmake to last commit hash + date
            return Version::PROJECT_VER_HASH;
        }

        static std::string timeStamp() {
            time_t time = std::chrono::system_clock::to_time_t(std::chrono::high_resolution_clock::now());
            return std::ctime(&time);
        }

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