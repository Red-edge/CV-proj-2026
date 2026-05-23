#include "central_control/common.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        auto config = central_control::parse_arguments(argc, argv);
        if (config.list_sources) {
            return central_control::list_available_sources(config);
        }
        central_control::Runtime runtime(std::move(config));
        if (!runtime.initialize()) {
            return 1;
        }
        return runtime.run();
    } catch (const std::exception& exc) {
        std::cerr << "central_control failed: " << exc.what() << std::endl;
        return 2;
    }
}
