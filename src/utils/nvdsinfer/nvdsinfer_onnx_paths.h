/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef NVDSINFER_ONNX_PATHS_H
#define NVDSINFER_ONNX_PATHS_H

// Internal publication guard, not an ONNX validator/converter. Read only the
// standard message paths that can contain TensorProto external-data locations;
// leave graph semantics and supported operators to the NVIDIA ONNX parser.
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvdsinfer_onnx_paths {
enum class Kind { Unknown, Model, Graph, Node, Attribute, Sparse, Function, Training, Tensor, Entry };
struct Edge { Kind parent; const char* name; unsigned number; Kind child; };
inline const std::vector<Edge>& edges() {
    // ONNX ModelProto/GraphProto/NodeProto/AttributeProto/SparseTensorProto/
    // FunctionProto/TrainingInfoProto/TensorProto field numbers, not wire guesses.
    static const std::vector<Edge> table = {
        {Kind::Model,"graph",7,Kind::Graph}, {Kind::Model,"training_info",20,Kind::Training},
        {Kind::Model,"functions",25,Kind::Function}, {Kind::Graph,"node",1,Kind::Node},
        {Kind::Graph,"initializer",5,Kind::Tensor}, {Kind::Graph,"sparse_initializer",15,Kind::Sparse},
        {Kind::Node,"attribute",5,Kind::Attribute}, {Kind::Attribute,"t",5,Kind::Tensor},
        {Kind::Attribute,"g",6,Kind::Graph}, {Kind::Attribute,"tensors",10,Kind::Tensor},
        {Kind::Attribute,"graphs",11,Kind::Graph}, {Kind::Attribute,"sparse_tensor",22,Kind::Sparse},
        {Kind::Attribute,"sparse_tensors",23,Kind::Sparse}, {Kind::Sparse,"values",1,Kind::Tensor},
        {Kind::Sparse,"indices",2,Kind::Tensor}, {Kind::Function,"node",7,Kind::Node},
        {Kind::Function,"attribute_proto",11,Kind::Attribute},
        {Kind::Training,"initialization",1,Kind::Graph}, {Kind::Training,"algorithm",2,Kind::Graph},
        {Kind::Tensor,"external_data",13,Kind::Entry}};
    return table;
}
inline void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(std::string("Cannot inspect ONNX input paths: ") + message);
}
constexpr unsigned kMaxDepth = 128;
constexpr size_t kMaxString = 1024 * 1024;
struct Metadata { std::string key, value; bool external = false; std::vector<std::string> locations; };
inline void addLocation(Metadata& tensor, const Metadata& entry) {
    if (entry.key != "location") return;
    require(tensor.locations.empty(), "duplicate external tensor location");
    tensor.locations.push_back(entry.value);
}
struct Result {
    bool graph = false;
    std::set<std::string> locations;
    void finish(Kind kind, const Metadata& state) {
        if (kind != Kind::Tensor) return;
        require(!state.external || !state.locations.empty(), "external tensor has no location");
        for (const auto& location : state.locations) {
            require(!location.empty() && location.find('\0') == std::string::npos,
                    "empty or NUL-containing external location");
            require(locations.size() < 65536, "too many external locations");
            locations.insert(location);
        }
    }
};

class Binary {
    std::ifstream input;
    uint64_t position = 0, size = 0;
    Result result;
    unsigned byte(uint64_t end) {
        require(position < end, "truncated protobuf field");
        int value = input.get();
        require(value != std::char_traits<char>::eof(), "unreadable protobuf field");
        ++position;
        return static_cast<unsigned char>(value);
    }
    uint64_t varint(uint64_t end) {
        uint64_t value = 0;
        for (unsigned shift = 0; shift < 70; shift += 7) {
            const unsigned next = byte(end);
            require(shift != 63 || next <= 1, "overflowing protobuf varint");
            value |= uint64_t(next & 127) << shift;
            if (!(next & 128)) return value;
        }
        throw std::runtime_error("Cannot inspect ONNX input paths: malformed protobuf varint");
    }
    uint64_t lengthEnd(uint64_t end) {
        const uint64_t length = varint(end);
        require(length <= end - position, "truncated protobuf message");
        return position + length;
    }
    void skipBytes(uint64_t count, uint64_t end) {
        require(count <= end - position, "truncated protobuf payload");
        position += count;
        input.seekg(static_cast<std::streamoff>(position));
        require(bool(input), "unreadable protobuf payload");
    }
    std::string string(uint64_t end) {
        const uint64_t stop = lengthEnd(end), length = stop - position;
        require(length <= kMaxString, "external metadata string too long");
        std::string value(static_cast<size_t>(length), '\0');
        if (length) input.read(&value[0], static_cast<std::streamsize>(length));
        require(bool(input), "unreadable metadata string");
        position = stop;
        return value;
    }
    void skip(unsigned wire, unsigned field, uint64_t end, unsigned depth) {
        require(depth <= kMaxDepth, "protobuf nesting limit exceeded");
        if (wire == 0) { varint(end); return; }
        if (wire == 1 || wire == 5) { skipBytes(wire == 1 ? 8 : 4, end); return; }
        if (wire == 2) { const auto stop = lengthEnd(end); skipBytes(stop - position, end); return; }
        require(wire == 3, "unsupported protobuf wire type");
        while (position < end) {
            const auto tag = varint(end);
            require(tag && tag <= std::numeric_limits<uint32_t>::max(), "invalid protobuf tag");
            if ((tag & 7) == 4) {
                require((tag >> 3) == field, "mismatched protobuf group");
                return;
            }
            skip(tag & 7, tag >> 3, end, depth + 1);
        }
        require(false, "unterminated protobuf group");
    }
    Metadata walk(Kind kind, uint64_t end, unsigned depth) {
        require(depth <= kMaxDepth, "protobuf nesting limit exceeded");
        Metadata state;
        while (position < end) {
            const auto tag = varint(end);
            require(tag && tag <= std::numeric_limits<uint32_t>::max() && (tag >> 3), "invalid protobuf tag");
            const unsigned field = tag >> 3, wire = tag & 7;
            const Edge* edge = nullptr;
            for (const auto& item : edges()) if (item.parent == kind && item.number == field) { edge = &item; break; }
            if (edge) {
                require(wire == 2, "invalid external tensor message encoding");
                if (kind == Kind::Model && field == 7) result.graph = true;
                auto child = walk(edge->child, lengthEnd(end), depth + 1);
                if (kind == Kind::Tensor) addLocation(state, child);
            } else if (kind == Kind::Entry && (field == 1 || field == 2)) {
                require(wire == 2, "invalid external-data entry encoding");
                (field == 1 ? state.key : state.value) = string(end);
            } else if (kind == Kind::Tensor && field == 14) {
                require(wire == 0, "invalid external-data location encoding");
                state.external |= varint(end) == 1;
            } else skip(wire, field, end, depth + 1);
        }
        result.finish(kind, state);
        return state;
    }
public:
    explicit Binary(const std::string& path) : input(path, std::ios::binary | std::ios::ate) {
        const auto end = input.tellg();
        require(bool(input) && end > 0, "model is empty or unreadable");
        size = static_cast<uint64_t>(end);
        input.seekg(0);
    }
    Result read() { walk(Kind::Model, size, 0); require(result.graph, "model has no graph"); return result; }
};

// Tokenize only enough protobuf text syntax to find the same metadata messages.
// Unknown values are skipped, never interpreted as graph data or file paths.
class Text {
    struct Token { enum Type { End, Bare, String, Punctuation } type = End; std::string value; bool large = false; } token;
    std::ifstream input;
    Result result;
    static bool punctuation(int c) { return c == '{' || c == '}' || c == '<' || c == '>' ||
        c == '[' || c == ']' || c == ':' || c == ',' || c == ';'; }
    static int hex(int c) { return c >= '0' && c <= '9' ? c - '0' :
        c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }
    void whitespace() {
        for (;;) {
            while (input.peek() != EOF && std::isspace(static_cast<unsigned char>(input.peek()))) input.get();
            if (input.peek() == '#') { input.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); continue; }
            if (input.peek() != '/') return;
            input.get();
            if (input.peek() == '/') { input.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); continue; }
            if (input.peek() == '*') {
                input.get(); int previous = 0, current = 0;
                while ((current = input.get()) != EOF && !(previous == '*' && current == '/')) previous = current;
                require(current != EOF, "unterminated text comment");
                continue;
            }
            input.unget(); return;
        }
    }
    static void utf8(std::string& out, uint32_t value) {
        require(value <= 0x10ffff && !(value >= 0xd800 && value <= 0xdfff), "invalid Unicode escape");
        if (value < 0x80) out += char(value);
        else if (value < 0x800) { out += char(0xc0 | value >> 6); out += char(0x80 | (value & 63)); }
        else if (value < 0x10000) { out += char(0xe0 | value >> 12); out += char(0x80 | (value >> 6 & 63)); out += char(0x80 | (value & 63)); }
        else { out += char(0xf0 | value >> 18); out += char(0x80 | (value >> 12 & 63)); out += char(0x80 | (value >> 6 & 63)); out += char(0x80 | (value & 63)); }
    }
    std::string quoted(bool& large) {
        const int quote = input.get(); std::string out;
        for (;;) {
            int c = input.get();
            require(c != EOF && c != '\n' && c != '\r', "unterminated text string");
            if (c == quote) return out;
            if (c == '\\') {
                c = input.get(); require(c != EOF, "truncated text escape");
                const std::string escapes = "abfnrtv\\\"'", decoded = "\a\b\f\n\r\t\v\\\"'";
                const auto found = escapes.find(char(c));
                if (found != std::string::npos) out += decoded[found];
                else if (c >= '0' && c <= '7') {
                    unsigned value = c - '0';
                    for (int n = 1; n < 3 && input.peek() >= '0' && input.peek() <= '7'; ++n) value = value * 8 + input.get() - '0';
                    require(value <= 255, "overflowing octal escape"); out += char(value);
                } else if (c == 'x' || c == 'X' || c == 'u' || c == 'U') {
                    const unsigned digits = c == 'u' ? 4 : c == 'U' ? 8 : 2;
                    unsigned n = 0; uint32_t value = 0;
                    for (; n < digits && hex(input.peek()) >= 0; ++n) value = value * 16 + hex(input.get());
                    require(n && (digits == 2 || n == digits), "invalid hexadecimal/Unicode escape");
                    if (digits == 2) out += char(value); else utf8(out, value);
                } else require(false, "unknown text escape");
            } else out += char(c);
            // Raw inline tensor payloads may be very large. Scan their syntax
            // without retaining them; only metadata consumers require a bound.
            if (out.size() > kMaxString) { large = true; out.resize(kMaxString); }
        }
    }
    void next() {
        whitespace(); token = Token{};
        const int c = input.peek();
        if (c == EOF) { require(!input.bad(), "unreadable model text"); return; }
        if (c == '\'' || c == '"') { token.type = Token::String; token.value = quoted(token.large); return; }
        if (punctuation(c)) { token.type = Token::Punctuation; token.value = char(input.get()); return; }
        token.type = Token::Bare;
        while (input.peek() != EOF && !std::isspace(static_cast<unsigned char>(input.peek())) &&
               !punctuation(input.peek()) && input.peek() != '"' && input.peek() != '\'' &&
               input.peek() != '#' && input.peek() != '/') {
            require(static_cast<unsigned char>(input.peek()) >= 32, "not protobuf text");
            token.value += char(input.get()); require(token.value.size() <= kMaxString, "text token too long");
        }
        require(!token.value.empty(), "invalid text token");
    }
    bool at(char c) const { return token.type == Token::Punctuation && token.value[0] == c; }
    void separators() { while (at(',') || at(';')) next(); }
    std::string scalar() {
        require(token.type == Token::Bare || token.type == Token::String, "invalid text scalar");
        require(!token.large, "external metadata string too long");
        const bool strings = token.type == Token::String; std::string value = token.value; next();
        while (strings && token.type == Token::String) { require(!token.large, "external metadata string too long"); value += token.value; require(value.size() <= kMaxString, "text string too long"); next(); }
        return value;
    }
    void skip(unsigned depth) {
        require(depth <= kMaxDepth, "text nesting limit exceeded");
        if (at('{') || at('<') || at('[')) {
            const char close = at('{') ? '}' : at('<') ? '>' : ']'; next();
            while (!at(close)) {
                require(token.type != Token::End && !at('}') && !at('>') && !at(']'), "unbalanced text value");
                if (at(':') || at(',') || at(';')) next(); else skip(depth + 1);
            }
            next();
        } else if (token.type == Token::String) {
            do { next(); } while (token.type == Token::String);
        } else { scalar(); }
    }
    Metadata message(Kind kind, unsigned depth) {
        require(at('{') || at('<'), "expected text message");
        const char close = at('{') ? '}' : '>'; next();
        return walk(kind, close, depth + 1);
    }
    Metadata walk(Kind kind, char close, unsigned depth) {
        require(depth <= kMaxDepth, "text nesting limit exceeded"); Metadata state;
        for (;;) {
            separators();
            if (close && at(close)) { next(); break; }
            if (token.type == Token::End) { require(!close, "truncated text message"); break; }
            require(token.type == Token::Bare, "expected text field name");
            const std::string field = token.value; next(); if (at(':')) next();
            const Edge* edge = nullptr;
            for (const auto& item : edges()) if (item.parent == kind && field == item.name) { edge = &item; break; }
            if (edge) {
                if (kind == Kind::Model && field == "graph") result.graph = true;
                const bool list = at('['); if (list) next();
                while (true) {
                    separators(); if (list && at(']')) { next(); break; }
                    auto child = message(edge->child, depth);
                    if (kind == Kind::Tensor) addLocation(state, child);
                    if (!list) break;
                }
            } else if (kind == Kind::Entry && (field == "key" || field == "value")) {
                require(token.type == Token::String, "external text location must be quoted");
                (field == "key" ? state.key : state.value) = scalar();
            } else if (kind == Kind::Tensor && field == "data_location") {
                const auto value = scalar(); state.external |= value == "EXTERNAL" || value == "1";
            } else skip(depth + 1);
        }
        result.finish(kind, state); return state;
    }
public:
    explicit Text(const std::string& path) : input(path, std::ios::binary) { require(bool(input), "unreadable model text"); next(); }
    Result read() {
        if (at('{') || at('<')) message(Kind::Model, 0); else walk(Kind::Model, 0, 0);
        require(token.type == Token::End && result.graph, "model text has no graph or has trailing data"); return result;
    }
};

inline std::set<std::string> inputFiles(const std::string& model) {
    Result found; bool binary = false, text = false; std::string error;
    try { found = Binary(model).read(); binary = true; } catch (const std::exception& e) { error = e.what(); }
    // Run the text scanner as well when plausible. A text document must never
    // evade location checks merely because its bytes also resemble wire fields.
    try { const auto other = Text(model).read(); found.locations.insert(other.locations.begin(), other.locations.end()); text = true; }
    catch (const std::exception& e) { if (!binary) error += std::string("; ") + e.what(); }
    if (!binary && !text) throw std::runtime_error(error);
    const auto parent = [](const std::string& path) { const auto slash = path.rfind('/');
        return slash == std::string::npos ? std::string(".") : slash == 0 ? std::string("/") : path.substr(0, slash); };
    std::set<std::string> bases{parent(model)}, paths{model};
    std::unique_ptr<char, decltype(&std::free)> canonical(realpath(model.c_str(), nullptr), &std::free);
    if (canonical) { paths.insert(canonical.get()); bases.insert(parent(canonical.get())); }
    for (const auto& location : found.locations) {
        if (location[0] == '/') paths.insert(location);
        else for (const auto& base : bases) paths.insert(base + '/' + location);
    }
    // Retain real targets too: a symlink changed during construction must not
    // erase a file that belonged to the initial protected input closure.
    const auto unresolved = paths;
    for (const auto& path : unresolved) {
        std::unique_ptr<char, decltype(&std::free)> resolved(realpath(path.c_str(), nullptr), &std::free);
        if (resolved) paths.insert(resolved.get());
    }
    return paths;
}
} // namespace nvdsinfer_onnx_paths
#endif
