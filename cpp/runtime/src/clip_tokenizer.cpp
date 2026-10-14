#include "clip_tokenizer.hpp"
#include "sd15/runtime.hpp"

#include <nlohmann/json.hpp>
#include <unicode/locid.h>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace sd15::detail {
namespace {

constexpr int32_t kBos = 49406;
constexpr int32_t kEos = 49407;
constexpr char kBosText[] = "<|startoftext|>";
constexpr char kEosText[] = "<|endoftext|>";
constexpr char kWordEnd[] = "</w>";
constexpr std::size_t kMaxPromptBytes = 16384;
constexpr uintmax_t kMaxAssetBytes = 4 * 1024 * 1024;
constexpr char kTokenPattern[] =
    R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+)";

void check_unicode(UErrorCode status) {
    if (U_FAILURE(status)) {
        throw std::runtime_error(std::string("ICU: ") + u_errorName(status));
    }
}

std::string utf8(UChar32 codepoint) {
    std::string result;
    icu::UnicodeString(codepoint).toUTF8String(result);
    return result;
}

std::ifstream open_asset(const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    if (size == 0 || size > kMaxAssetBytes) {
        throw std::invalid_argument("Invalid tokenizer asset size: " + path.string());
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot read tokenizer asset: " + path.string());
    }
    return stream;
}

icu::UnicodeString normalize(const std::string& prompt) {
    if (prompt.size() > kMaxPromptBytes) {
        throw std::invalid_argument("Prompt exceeds 16384 UTF-8 bytes");
    }
    const auto length = static_cast<int32_t>(prompt.size());
    for (int32_t i = 0; i < length;) {
        UChar32 codepoint;
        U8_NEXT(prompt.data(), i, length, codepoint);
        if (codepoint < 0) {
            throw std::invalid_argument("Prompt must be valid UTF-8");
        }
    }
    UErrorCode status = U_ZERO_ERROR;
    const auto* nfc = icu::Normalizer2::getNFCInstance(status);
    check_unicode(status);
    icu::UnicodeString normalized;
    nfc->normalize(icu::UnicodeString::fromUTF8(prompt), normalized, status);
    check_unicode(status);

    icu::UnicodeString result;
    bool whitespace = false;
    for (int32_t i = 0; i < normalized.length();) {
        const UChar32 codepoint = normalized.char32At(i);
        i += U16_LENGTH(codepoint);
        if (u_isUWhiteSpace(codepoint)) {
            if (!whitespace) {
                result.append(static_cast<UChar32>(' '));
            }
            whitespace = true;
            continue;
        }
        whitespace = false;
        // HF tokenizers는 문맥 없이 문자별 full lowercase를 적용한다. Case folding과 다르다.
        icu::UnicodeString letter(codepoint);
        result.append(letter.toLower(icu::Locale::getRoot()));
    }
    return result;
}

} // namespace

ClipTokenizer::ClipTokenizer(const std::string& directory) {
    const std::filesystem::path root(directory);
    auto vocabulary = open_asset(root / "vocab.json");
    const auto json = nlohmann::json::parse(vocabulary);
    if (!json.is_object() || json.size() != kVocabularySize) {
        throw std::invalid_argument("Expected SD1.5 CLIP vocabulary");
    }
    std::vector<bool> seen(kVocabularySize);
    for (auto it = json.begin(); it != json.end(); ++it) {
        if (!it->is_number_integer() || *it < 0 || *it >= kVocabularySize) {
            throw std::invalid_argument("Invalid CLIP vocabulary ID");
        }
        const int32_t id = it->get<int32_t>();
        if (seen[id]) {
            throw std::invalid_argument("Duplicate CLIP vocabulary ID");
        }
        seen[id] = true;
        vocabulary_.emplace(it.key(), id);
    }
    if (vocabulary_.at(kBosText) != kBos || vocabulary_.at(kEosText) != kEos) {
        throw std::invalid_argument("Unexpected CLIP special tokens");
    }

    // CLIP의 byte-to-Unicode 매핑. 단어의 마지막 byte에는 </w>를 붙인다.
    int extra = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const bool direct = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        bytes_[byte] = utf8(direct ? byte : extra++);
        if (!vocabulary_.count(bytes_[byte]) || !vocabulary_.count(bytes_[byte] + kWordEnd)) {
            throw std::invalid_argument("Missing CLIP byte token");
        }
    }
    auto merges = open_asset(root / "merges.txt");
    std::string line;
    if (!std::getline(merges, line) || line.rfind("#version: 0.2", 0) != 0) {
        throw std::invalid_argument("Unsupported CLIP merges format");
    }
    int rank = 0;
    while (std::getline(merges, line)) {
        if (line.empty() || line == "\r") {
            continue;
        }
        std::istringstream fields(line);
        std::string first, second, extra_field;
        if (!(fields >> first >> second) || (fields >> extra_field) || !vocabulary_.count(first) ||
            !vocabulary_.count(second) || !vocabulary_.count(first + second) ||
            !ranks_.emplace(first + '\n' + second, rank++).second) {
            throw std::invalid_argument("Invalid CLIP merge");
        }
    }
    if (ranks_.empty() || merges.bad()) {
        throw std::runtime_error("Cannot read CLIP merges");
    }
}

std::vector<int32_t> ClipTokenizer::encode(const std::string& prompt) const {
    std::vector<int32_t> result{kBos};
    const auto text = normalize(prompt);
    static const auto pattern = [] {
        UErrorCode status = U_ZERO_ERROR;
        std::unique_ptr<icu::RegexPattern> compiled(
            icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(kTokenPattern), 0, status));
        check_unicode(status);
        return compiled;
    }();
    UErrorCode status = U_ZERO_ERROR;
    std::unique_ptr<icu::RegexMatcher> matcher(pattern->matcher(text, status));
    check_unicode(status);
    while (matcher->find(status)) {
        if (result.size() >= kSequenceLength - 1) {
            break;
        }
        std::string token;
        matcher->group(status).toUTF8String(token);
        check_unicode(status);
        if (token == kBosText || token == kEosText) {
            result.push_back(vocabulary_.at(token));
            continue;
        }
        std::vector<std::string> pieces;
        for (unsigned char byte : token) {
            pieces.push_back(bytes_[byte]);
        }
        if (pieces.empty()) {
            continue;
        }
        pieces.back() += kWordEnd;
        while (pieces.size() > 1) {
            int rank = std::numeric_limits<int>::max();
            std::size_t selected = pieces.size();
            for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                const auto found = ranks_.find(pieces[i] + '\n' + pieces[i + 1]);
                if (found != ranks_.end() && found->second < rank) {
                    rank = found->second;
                    selected = i;
                }
            }
            if (selected == pieces.size()) {
                break;
            }
            pieces[selected] += pieces[selected + 1];
            pieces.erase(pieces.begin() + selected + 1);
        }
        for (const auto& piece : pieces) {
            if (result.size() >= kSequenceLength - 1) {
                break;
            }
            result.push_back(vocabulary_.at(piece));
        }
    }
    check_unicode(status);
    result.push_back(kEos);
    result.resize(kSequenceLength, kEos);
    return result;
}

} // namespace sd15::detail
