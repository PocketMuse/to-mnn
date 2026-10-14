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

// Transformers 5.17.0 / tokenizers 0.23.2, 동일 SD1.5 vocab·merges의 기준 ID.
TEST_CASE("CLIP Unicode preprocessing matches Hugging Face") {
    sd15::detail::ClipTokenizer tokenizer(tokenizer_dir);
    const auto check = [&](const std::string& text, std::vector<int32_t> expected) {
        CAPTURE(text);
        expected.resize(sd15::kSequenceLength, 49407);
        CHECK(tokenizer.encode(text) == expected);
    };
    check("a photo of a cat", {49406, 320, 1125, 539, 320, 2368, 49407});
    check("한글 고양이 猫", {49406, 15197, 250, 32495, 478, 36216, 254, 31625, 239, 34746, 163, 234, 360, 49407});
    check("Café", {49406, 15304, 49407});
    check("Cafe\u0301", {49406, 15304, 49407});
    check("가", {49406, 21122, 478, 49407});
    check("\u1100\u1161", {49406, 21122, 478, 49407});
    check("İstanbul", {49406, 328, 16384, 11231, 49407});
    check("ΟΣ", {49406, 138, 123, 139, 481, 49407});
    check("a\u00a0cat", {49406, 320, 2368, 49407});
    check("a\u202fcat", {49406, 320, 2368, 49407});
    check("a\u200bcat", {49406, 320, 9844, 2368, 49407});
    check("\ufeffcat", {49406, 171, 119, 379, 2368, 49407});
    check("ﬁsh", {49406, 171, 105, 223, 849, 49407});
    check("ＡＢＣ", {49406, 171, 121, 223, 171, 121, 224, 171, 121, 481, 49407});
    check("cat &amp; dog", {49406, 2368, 261, 6259, 282, 1929, 49407});
    check("FranÃ§ais", {49406, 2118, 127, 352, 126, 356, 11393, 49407});
    check("a\034cat", {49406, 320, 472, 2368, 49407});
    check("🐈", {49406, 2074, 486, 49407});
    check("<|STARTOFTEXT|>cat", {49406, 49406, 2368, 49407});
    check("a\u0327\u0301", {49406, 22229, 136, 356, 49407});
    check("a\302\205cat", {49406, 320, 2368, 49407});
    check("a\u2028cat", {49406, 320, 2368, 49407});
    check("a\u2029cat", {49406, 320, 2368, 49407});
    check("STRAẞE", {49406, 1894, 127, 253, 324, 49407});
    check("ος", {49406, 138, 123, 139, 480, 49407});
    check("οσ", {49406, 138, 123, 139, 481, 49407});
}

TEST_CASE("CLIP rejects oversized and invalid UTF-8 prompts") {
    sd15::detail::ClipTokenizer tokenizer(tokenizer_dir);
    REQUIRE_THROWS_AS(tokenizer.encode(std::string(16385, 'a')), std::invalid_argument);
    for (const auto& invalid : {std::string("\xff", 1), std::string("\xc0\xaf", 2), std::string("\xed\xa0\x80", 3),
                                std::string("\xf4\x90\x80\x80", 4), std::string("\xe2\x82", 2)}) {
        CHECK_THROWS_AS(tokenizer.encode(invalid), std::invalid_argument);
    }
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
