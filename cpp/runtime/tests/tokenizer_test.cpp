#include "clip_tokenizer.hpp"
#include "sd15/runtime.hpp"

#include <iostream>
#include <iterator>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) {
            throw std::invalid_argument("Expected tokenizer directory");
        }
        sd15::detail::ClipTokenizer tokenizer(argv[1]);
        if (argc == 3) {
            const std::string prompt =
                std::string(argv[2]) == "--stdin" ? std::string(std::istreambuf_iterator<char>(std::cin), {}) : argv[2];
            for (const auto id : tokenizer.encode(prompt)) {
                std::cout << id << ' ';
            }
            return 0;
        }
        const auto check = [&](const std::string& text, std::vector<int32_t> expected) {
            expected.resize(sd15::kSequenceLength, 49407);
            if (tokenizer.encode(text) != expected) {
                for (const auto id : tokenizer.encode(text)) {
                    std::cerr << id << ' ';
                }
                std::cerr << '\n';
                throw std::runtime_error("Token mismatch: " + text);
            }
        };
        check("", {49406, 49407});
        check("a photo of a cat", {49406, 320, 1125, 539, 320, 2368, 49407});
        check(" A PHOTO\t of\n a CAT ", {49406, 320, 1125, 539, 320, 2368, 49407});
        for (const auto& invalid : {std::string(16385, 'a'), std::string("\xff", 1)}) {
            bool rejected = false;
            try {
                tokenizer.encode(invalid);
            }
            catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected) {
                throw std::runtime_error("Invalid prompt was accepted");
            }
        }
        std::string long_prompt;
        for (int i = 0; i < 100; ++i) {
            long_prompt += "cat ";
        }
        auto expected = std::vector<int32_t>(sd15::kSequenceLength, 2368);
        expected.front() = 49406;
        expected.back() = 49407;
        if (tokenizer.encode(long_prompt) != expected) {
            throw std::runtime_error("Truncation must preserve EOS");
        }
        std::cout << "CLIP token IDs and truncation passed\n";
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
