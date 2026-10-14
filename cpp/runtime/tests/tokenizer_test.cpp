#include "clip_tokenizer.hpp"
#include "sd15/runtime.hpp"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
std::string tokenizer_dir;
}

TEST_CASE("CLIP token IDs preserve whitespace normalization and special tokens") {
    sd15::detail::ClipTokenizer tokenizer(tokenizer_dir);
    const auto check = [&](const std::string& text, std::vector<int32_t> expected) {
        CAPTURE(text);
        expected.resize(sd15::kSequenceLength, 49407);
        CHECK(tokenizer.encode(text) == expected);
    };
    check("", {49406, 49407});
    check("a photo of a cat", {49406, 320, 1125, 539, 320, 2368, 49407});
    check(" A PHOTO\t of\n a CAT ", {49406, 320, 1125, 539, 320, 2368, 49407});
}

TEST_CASE("CLIP rejects oversized and invalid UTF-8 prompts") {
    sd15::detail::ClipTokenizer tokenizer(tokenizer_dir);
    REQUIRE_THROWS_AS(tokenizer.encode(std::string(16385, 'a')), std::invalid_argument);
    REQUIRE_THROWS_AS(tokenizer.encode(std::string("\xff", 1)), std::invalid_argument);
}

TEST_CASE("CLIP truncation preserves EOS") {
    sd15::detail::ClipTokenizer tokenizer(tokenizer_dir);
    std::string long_prompt;
    for (int i = 0; i < 100; ++i) {
        long_prompt += "cat ";
    }
    auto expected = std::vector<int32_t>(sd15::kSequenceLength, 2368);
    expected.front() = 49406;
    expected.back() = 49407;
    CHECK(tokenizer.encode(long_prompt) == expected);
}

int main(int argc, char** argv) {
    Catch::Session session;
    bool from_stdin = false;
    session.cli(
        session.cli() |
        Catch::Clara::Opt(tokenizer_dir, "directory")["--tokenizer-dir"]("CLIP tokenizer directory").required() |
        Catch::Clara::Opt(from_stdin)["--stdin"]("Print token IDs for a UTF-8 prompt from stdin"));
    const auto parsed = session.applyCommandLine(argc, argv);
    if (parsed != 0) {
        return parsed;
    }
    if (!from_stdin) {
        return session.run();
    }
    // Python 대조 테스트에는 Catch2 보고서 없이 token IDs만 전달한다.
    try {
        sd15::detail::ClipTokenizer tokenizer(tokenizer_dir);
        const std::string prompt(std::istreambuf_iterator<char>(std::cin), {});
        for (const auto id : tokenizer.encode(prompt)) {
            std::cout << id << ' ';
        }
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
